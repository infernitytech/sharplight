using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Import.Onnx;

// Recognition of attention written out with elementary ops (encoder exports: MatMul → [scale] → [mask Add] → Softmax → MatMul).
internal sealed partial class OnnxTranslator
{
    /// <summary>
    /// Translates <paramref name="n"/> = <c>MatMul(P, V)</c> as one fused <c>Attention</c> when P is
    /// <c>Softmax(BatchMatMul(Q, Kᵀ) [+ mask])</c> over the last axis, read only by <paramref name="n"/>. The products may be
    /// rank 4 ([B, H, S, D]) or rank 3 ([B·H, S, D], split back into B and H when the operands are reshaped from rank 4); a constant
    /// scale is already folded into the product's alpha (<see cref="TryScaledBatchMatMul"/>). Masks may be a key-padding bias
    /// broadcast over heads and queries ([B, 1, 1, T]) or, at batch 1, one [S, T] mask for all heads. The batch runs as the
    /// batched form of Attention (<see cref="BatchedAttention"/>). The elementary nodes already emitted are left to dead-node
    /// elimination. Returns false when the pattern does not match or fusion is off (<see cref="OnnxImportOptions.FuseAttention"/>).
    /// </summary>
    private bool TryDecomposedAttention(OnnxNode n, TranslatedTensor probabilities, TranslatedTensor values)
    {
        if (!_opts.FuseAttention
            || probabilities is not { LeadingUnits: 0, V: { Producer: { Op.Name: "Softmax" } softmax } }
            || values is not { LeadingUnits: 0, V: { } v }
            || softmax.Attributes.Get("log", false)
            || Producer(n.Inputs[0], "Softmax") is null
            || SoleConsumer(n.Inputs[0], "MatMul") != n)
        {
            return false;
        }

        var scores = softmax.Inputs[0];
        float softmaxScale = softmax.Attributes.Get("scale", 1f);
        Value? mask = null;
        Node? product = scores.Producer;
        if (product is { Op.Name: "Add" } add)
        {
            int productSide = add.Inputs[0].Producer is { Op.Name: "BatchMatMul" } && add.Inputs[0].Shape.Equals(scores.Shape) ? 0
                : add.Inputs[1].Producer is { Op.Name: "BatchMatMul" } && add.Inputs[1].Shape.Equals(scores.Shape) ? 1 : -1;
            if (productSide < 0 || softmaxScale != 1f)
            {
                return false;
            }

            mask = add.Inputs[1 - productSide];
            product = add.Inputs[productSide].Producer;
        }

        if (product is not { Op.Name: "BatchMatMul" } bmm || bmm.Attributes.Get("trans_a", false) || scores.Type != DataType.F32 || v.Type != DataType.F32)
        {
            return false;
        }

        var a = bmm.Inputs[0];
        var b = bmm.Inputs[1];
        int rank = scores.Rank;
        if (rank is not (3 or 4) || a.Rank != rank || b.Rank != rank || v.Rank != rank || (mask is not null && mask.Type != DataType.F32))
        {
            return false;
        }

        var keys = bmm.Attributes.Get("trans_b", false) ? b : PermuteView(b, rank == 4 ? [0, 1, 3, 2] : [0, 2, 1]);
        var batchDims = scores.Shape.Dims[..^2];
        if (!a.Shape.Dims[..^2].SequenceEqual(batchDims) || !keys.Shape.Dims[..^2].SequenceEqual(batchDims) || !v.Shape.Dims[..^2].SequenceEqual(batchDims)
            || a.Shape[-1] != keys.Shape[-1] || keys.Shape[-2] != v.Shape[-2] || a.Shape[-1].IsSymbolic || v.Shape[-1].IsSymbolic
            || a.Shape[-2].IsSymbolic || keys.Shape[-2].IsSymbolic)
        {
            return false;
        }

        long s = a.Shape[-2].Fixed;
        long t = keys.Shape[-2].Fixed;
        long d = a.Shape[-1].Fixed;
        long dv = v.Shape[-1].Fixed;
        if (d % 8 != 0 || dv % 8 != 0)
        {
            return false;
        }

        // Rank 4 [B, H, S, D] operands; rank-3 [B·H, S, D] ones are split back into B and H when all three are reshaped from rank 4.
        Value q4;
        Value k4;
        Value v4;
        if (rank == 4)
        {
            (q4, k4, v4) = (a, keys, v);
        }
        else if (Unmerged(a) is { } qu && Unmerged(keys) is { } ku && Unmerged(v) is { } vu
                 && qu.Shape[0] == ku.Shape[0] && qu.Shape[1] == ku.Shape[1] && qu.Shape[0] == vu.Shape[0] && qu.Shape[1] == vu.Shape[1])
        {
            (q4, k4, v4) = (qu, ku, vu);
        }
        else
        {
            // One head per batch entry.
            (q4, k4, v4) = (ToHeads(a, a.Shape[0], 1), ToHeads(keys, keys.Shape[0], 1), ToHeads(v, v.Shape[0], 1));
        }

        if ((rank != 4 && mask is not null) || FusedAttention(q4, k4, v4, bmm.Attributes.Get("alpha", 1f) * softmaxScale, mask) is not { } y4)
        {
            return false;
        }

        SetDyn(n, 0, rank == 4 ? y4 : ReshapeOuter(y4, y4.Shape[0] * y4.Shape[1], s, dv));
        return true;
    }

    /// <summary>
    /// Bidirectional attention of q [B, Hq, S, D] over k [B, Hkv, T, D] / v [B, Hkv, T, Dv] with an optional additive
    /// <paramref name="mask"/> broadcastable to [B, Hq, S, T], as one fused Attention op (the batched form for B &gt; 1). Returns the
    /// output [B, Hq, S, Dv], or null when the shapes or the mask do not fit the fused op (nothing is emitted then, or only
    /// views that dead-node elimination removes).
    /// </summary>
    private Value? FusedAttention(Value q4, Value k4, Value v4, float scale, Value? mask)
    {
        if (!_opts.FuseAttention || q4.Rank != 4 || k4.Rank != 4 || v4.Rank != 4
            || q4.Shape.Dims[1..].Any(x => x.IsSymbolic) || k4.Shape.Dims[1..].Any(x => x.IsSymbolic) || v4.Shape.Dims[1..].Any(x => x.IsSymbolic)
            || q4.Shape[0] != k4.Shape[0] || q4.Shape[0] != v4.Shape[0] || k4.Shape[1] != v4.Shape[1] || k4.Shape[2] != v4.Shape[2]
            || q4.Shape[3] != k4.Shape[3] || q4.Type != DataType.F32 || k4.Type != DataType.F32 || v4.Type != DataType.F32)
        {
            return null;
        }

        var batch = q4.Shape[0];
        long heads = q4.Shape[1].Fixed;
        long kvHeads = k4.Shape[1].Fixed;
        long s = q4.Shape[2].Fixed;
        long t = k4.Shape[2].Fixed;
        long d = q4.Shape[3].Fixed;
        long dv = v4.Shape[3].Fixed;
        if (heads % kvHeads != 0 || d % 8 != 0 || dv % 8 != 0 || (mask is not null && mask.Type != DataType.F32))
        {
            return null;
        }

        Value? maskOperand = null;
        if (mask is not null && (maskOperand = AttentionMaskOperand(mask, batch, s, t)) is null)
        {
            return null;
        }

        var attributes = new List<(string, object)> { ("scale", scale), ("causal", false) };
        if (batch != Dim.Of(1))
        {
            attributes.Add((BatchedAttention.RowsAttribute, s));
            attributes.Add((BatchedAttention.KeysAttribute, t));
        }

        Value[] operands = [Rows(q4, batch, s, heads, d), Rows(k4, batch, t, kvHeads, d), Rows(v4, batch, t, kvHeads, dv)];
        if (maskOperand is not null)
        {
            operands = [.. operands, maskOperand];
        }

        var y = Emit("Attention", operands, Attributes.From([.. attributes]));
        Heavy();

        // [B·S, H, Dv] → [B, S, H, Dv] → [B, H, S, Dv].
        return PermuteView(ReshapeOuter(y, batch, s, heads, dv), [0, 2, 1, 3]);
    }

    /// <summary>
    /// The mask operand of the fused attention for an additive mask broadcast to scores [B, H, S, T]: a key bias [B·T] when the
    /// mask only varies by batch entry and key, or the [S, T] mask at batch 1; null when the mask has another shape.
    /// </summary>
    private Value? AttentionMaskOperand(Value mask, Dim batch, long s, long t)
    {
        var dims = Enumerable.Repeat(Dim.Of(1), 4 - mask.Rank).Concat(mask.Shape.Dims).ToArray();
        if (mask.Rank > 4 || dims[1] != Dim.Of(1) || dims[3] != Dim.Of(t))
        {
            return null;
        }

        if (dims[2] == Dim.Of(1) && (dims[0] == batch || (dims[0] == Dim.Of(1) && batch == Dim.Of(1))))
        {
            return ReshapeOuter(mask, batch * Dim.Of(t));
        }

        return dims[2] == Dim.Of(s) && dims[0] == Dim.Of(1) && batch == Dim.Of(1) ? Rs(mask, s, t) : null;
    }

    /// <summary>[B, H, R, D] → the attention row layout [B·R, H, D] (a view when the source is [B, R, H, D] in memory).</summary>
    private Value Rows(Value x4, Dim batch, long rows, long heads, long d)
        => heads == 1 ? ReshapeOuter(x4, batch * Dim.Of(rows), 1, d) : ReshapeOuter(PermuteView(x4, [0, 2, 1, 3]), batch * Dim.Of(rows), heads, d);

    /// <summary>[B·H, R, D] (with H = <paramref name="heads"/>) as [B, H, R, D].</summary>
    private Value ToHeads(Value x3, Dim batch, long heads) => ReshapeOuter(x3, batch, heads, x3.Shape[1].Fixed, x3.Shape[2].Fixed);

    /// <summary>The rank-4 [B, H, R, D] value a rank-3 [B·H, R, D] operand was reshaped from, or null.</summary>
    private static Value? Unmerged(Value x3)
    {
        if (x3.Producer is not { Op.Name: "Reshape" } reshape || reshape.Inputs[0] is not { Rank: 4 } u)
        {
            return null;
        }

        return u.Shape[2] == x3.Shape[1] && u.Shape[3] == x3.Shape[2] && !u.Shape[1].IsSymbolic && u.Shape[0] * u.Shape[1] == x3.Shape[0] ? u : null;
    }

    /// <summary>A permutation of <paramref name="x"/> that composes with a permutation producing it (and vanishes when the two cancel).</summary>
    private Value PermuteView(Value x, int[] perm)
    {
        if (x.Producer is { Op.Name: "Permute" } inner)
        {
            var innerPerm = inner.Attributes.Get<int[]>("perm");
            int[] composed = [.. perm.Select(p => innerPerm[p])];
            return Permute(inner.Inputs[0], composed);
        }

        return Permute(x, perm);
    }

    /// <summary>Reshape to [<paramref name="outer"/>, ..<paramref name="inner"/>] where the outer dimension may be symbolic.</summary>
    private Value ReshapeOuter(Value x, Dim outer, params long[] inner)
    {
        if (!outer.IsSymbolic)
        {
            return Rs(x, [outer.Fixed, .. inner]);
        }

        if (x.Shape[0] == outer)
        {
            return Emit("Reshape", [x], Attributes.From(("shape", (long[])[0, .. inner])));
        }

        return Emit("Reshape", [x], Attributes.From(("shape", (long[])[0, .. inner]), ("outer_symbol", outer.Symbol!)));
    }
}
