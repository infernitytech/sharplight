using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Onnx;

// Matrix products: Gemm, MatMul, Einsum and block-quantized MatMulNBits.
internal sealed partial class OnnxTranslator
{
    private void Gemm(OnnxNode n, TranslatedTensor?[] ins)
    {
        float alpha = n.Float("alpha", 1f);
        float beta = n.Float("beta", 1f);
        bool tA = n.Int("transA", 0) != 0;
        bool tB = n.Int("transB", 0) != 0;
        var a = X(ins, 0);
        Value y;
        string name = Nm(n);
        if (ins[1]!.IsConst && ins[1]!.C!.Rank == 2)
        {
            if (tA)
            {
                a = Permute(a, [1, 0]);
            }

            var bc = ins[1]!.C!;
            var w = WeightValue(tB ? bc : bc.Transposed([1, 0]), keepHalf: true, name + ".weight");
            y = Emit("MatMul", [a, w], null, name);
            if (alpha != 1f)
            {
                y = _b.Scale(y, alpha);
            }
        }
        else
        {
            y = Emit("BatchMatMul", [a, X(ins, 1)], Attributes.From(("trans_a", tA), ("trans_b", tB), ("alpha", alpha)), name);
        }
        Heavy();
        if (ins.Length > 2 && ins[2] is { } c)
        {
            Value cv;
            if (c.IsConst)
            {
                cv = ConstValue(beta == 1f ? c.C! : OnnxConst.FromFloats(OnnxType.Float, c.C!.Dims, c.C.Floats().Select(v => v * beta).ToArray(), c.C.Name + ".beta"), name + ".bias");
            }
            else
            {
                cv = beta == 1f ? c.V! : _b.Scale(c.V!, beta);
            }

            y = Emit("Add", [y, cv], null, name + ".biased");
        }
        SetDyn(n, 0, y);
    }

    private void MatMul(OnnxNode n, TranslatedTensor?[] ins)
    {
        var aT = ins[0]!;
        var bT = ins[1]!;
        string name = Nm(n);
        Value y;
        if (!aT.IsConst && bT.IsConst && bT.C!.Rank == 2 && aT.V!.Rank >= 2)
        {
            var w = WeightValue(bT.C!.Transposed([1, 0]), keepHalf: true, name + ".weight");
            y = Emit("MatMul", [aT.V!, w], null, name);
        }
        else
        {
            var a = AsValue(aT);
            var b = AsValue(bT);
            if (a.Rank < 2 || b.Rank < 2)
            {
                y = VectorMatMul(a, b, name);
            }
            else
            {
                y = Emit("BatchMatMul", [a, b], null, name);
            }
        }
        Heavy();
        SetDyn(n, 0, y);
    }

    /// <summary>MatMul with a 1-D operand (numpy semantics): the vector becomes a [1, K] row or [K, 1] column and that unit axis is dropped again.</summary>
    private Value VectorMatMul(Value a, Value b, string name)
    {
        bool rowVector = a.Rank == 1;
        bool columnVector = b.Rank == 1;
        var product = Emit("BatchMatMul", [rowVector ? Rs(a, 1, a.Shape[0].Fixed) : a, columnVector ? Rs(b, b.Shape[0].Fixed, 1) : b]);
        var dims = new List<long>();
        for (int i = 0; i < product.Rank; i++)
        {
            if ((rowVector && i == product.Rank - 2) || (columnVector && i == product.Rank - 1))
            {
                continue;
            }

            dims.Add(i == 0 && Sym(product) ? 0 : product.Shape[i].Fixed);
        }

        return Emit("Reshape", [product], Attributes.From(("shape", dims.ToArray())), name);
    }

    private Value MatMulWeightOrBatch(Value a, OnnxConst bf, string name)
    {
        Value y;
        if (bf.Rank == 2 && a.Rank >= 2)
        {
            y = Emit("MatMul", [a, WeightValue(bf.Transposed([1, 0]), false, name + ".weight")], null, name);
        }
        else
        {
            y = Emit("BatchMatMul", [a, ConstValue(bf, name + ".b")], null, name);
        }

        Heavy();
        return y;
    }

    private void Einsum(OnnxNode n, TranslatedTensor?[] ins)
    {
        string eq = n.Str("equation", "").Replace(" ", "");
        if (eq.Contains("..."))
        {
            throw new NotSupportedException("Einsum with ellipsis");
        }

        var parts = eq.Split("->");
        var terms = parts[0].Split(',');
        string rhs = parts.Length > 1 ? parts[1] : new string(string.Concat(terms).GroupBy(c => c).Where(g => g.Count() == 1).Select(g => g.Key).OrderBy(c => c).ToArray());
        if (terms.Length is < 1 or > 2)
        {
            throw new NotSupportedException("Einsum with more than two operands");
        }

        var ops = terms.Select((t, i) => new LabeledValue(X(ins, i), [.. t])).ToList();
        foreach (var o in ops)
        {
            if (o.L.Count != o.V.Rank || o.L.Distinct().Count() != o.L.Count)
            {
                throw new NotSupportedException("Einsum with repeated labels (traces/diagonals)");
            }
        }

        var outL = rhs.ToList();
        // sum out labels that appear in only one operand and not in the output
        for (int i = 0; i < ops.Count; i++)
        {
            var other = ops.Count == 2 ? ops[1 - i].L : [];
            var drop = ops[i].L.Where(c => !outL.Contains(c) && !other.Contains(c)).ToList();
            if (drop.Count > 0)
            {
                var axes = drop.Select(c => (long)ops[i].L.IndexOf(c)).ToArray();
                ops[i] = new LabeledValue(Emit("Reduce", [ops[i].V], Attributes.From(("mode", "sum"), ("axes", axes), ("keepdims", false))), ops[i].L.Where(c => !drop.Contains(c)).ToList());
            }
        }

        var result = ops.Count == 1 ? ops[0] : EinsumContract(ops[0], ops[1], outL);
        var final = PermuteLabels(result, outL);
        _env[n.Outputs[0]] = TranslatedTensor.Dyn(ReferenceEquals(final, result.V) ? Emit("Reshape", [final], Attributes.From(("shape", Fixed(final))), Nm(n)) : final);
    }

    /// <summary>
    /// Two-operand Einsum as one BatchMatMul: labels shared with the output are batch labels, labels shared but not in the output are
    /// contracted. The result is labelled [batch..., a-only..., b-only...].
    /// </summary>
    private LabeledValue EinsumContract(LabeledValue a, LabeledValue b, List<char> outL)
    {
        var batch = a.L.Where(c => b.L.Contains(c) && outL.Contains(c)).ToList();
        var contracted = a.L.Where(c => b.L.Contains(c) && !outL.Contains(c)).ToList();
        var aOnly = a.L.Where(c => !b.L.Contains(c)).ToList();
        var bOnly = b.L.Where(c => !a.L.Contains(c)).ToList();
        Value aPermuted = PermuteLabels(a, [.. batch, .. aOnly, .. contracted]);
        Value bPermuted = PermuteLabels(b, [.. batch, .. contracted, .. bOnly]);
        var dimOf = new Dictionary<char, Dim>();
        foreach (var operand in new[] { a, b })
        {
            for (int i = 0; i < operand.L.Count; i++)
            {
                dimOf[operand.L[i]] = operand.V.Shape[i];
            }
        }

        // A keeps its free dims separate (so a symbolic batch can stay in front); B gets matching size-1 dims to broadcast over them.
        var aGroups = new List<List<char>>(batch.Select(c => new List<char> { c }));
        if (aOnly.Count == 0)
        {
            aGroups.Add([]);
        }
        else
        {
            aGroups.AddRange(aOnly.Select(c => new List<char> { c }));
        }

        aGroups.Add(contracted);
        var bGroups = new List<List<char>>(batch.Select(c => new List<char> { c }));
        for (int i = 1; i < aOnly.Count; i++)
        {
            bGroups.Add([]);
        }

        bGroups.Add(contracted);
        bGroups.Add(bOnly);
        var aShaped = ReshapeGroups(aPermuted, aGroups, dimOf);
        var bShaped = ReshapeGroups(bPermuted, bGroups, dimOf);
        var product = Emit("BatchMatMul", [aShaped, bShaped]);
        Heavy();
        List<char> resultLabels = [.. batch, .. aOnly, .. bOnly];
        return new LabeledValue(ReshapeToLabels(product, resultLabels, dimOf), resultLabels);
    }

    /// <summary>An Einsum operand: an IR value and the subscript label of each of its dimensions.</summary>
    private sealed record LabeledValue(Value V, List<char> L);

    private Value PermuteLabels(LabeledValue t, List<char> order)
    {
        var perm = order.Select(c => t.L.IndexOf(c)).ToArray();
        return Permute(t.V, perm);
    }

    private Value ReshapeGroups(Value x, List<List<char>> groups, Dictionary<char, Dim> dimOf)
    {
        var shape = new List<long>();
        for (int gi = 0; gi < groups.Count; gi++)
        {
            long prod = 1;
            bool sym = false;
            foreach (var c in groups[gi])
            {
                if (dimOf[c].IsSymbolic)
                {
                    sym = true;
                }
                else
                {
                    prod *= dimOf[c].Fixed;
                }
            }
            if (sym)
            {
                if (shape.Count != 0 || groups[gi].Count != 1)
                {
                    throw new NotSupportedException("Einsum: the batch dimension would be merged or moved");
                }

                shape.Add(0);
            }
            else
            {
                shape.Add(prod);
            }
        }
        return Emit("Reshape", [x], Attributes.From(("shape", shape.ToArray())));
    }

    private Value ReshapeToLabels(Value y, List<char> labels, Dictionary<char, Dim> dimOf)
    {
        var shape = new List<long>();
        for (int i = 0; i < labels.Count; i++)
        {
            if (dimOf[labels[i]].IsSymbolic)
            {
                if (i != 0)
                {
                    throw new NotSupportedException("Einsum: batch dimension in a non-leading position");
                }
                shape.Add(0);
            }
            else
            {
                shape.Add(dimOf[labels[i]].Fixed);
            }
        }
        return Emit("Reshape", [y], Attributes.From(("shape", shape.ToArray())));
    }

    private void MatMulNBits(OnnxNode n, TranslatedTensor?[] ins)
    {
        long K = n.Int("K", 0);
        long N = n.Int("N", 0);
        long bits = n.Int("bits", 4);
        long bs = n.Int("block_size", 32);
        if (bits is not (4 or 8))
        {
            throw new NotSupportedException($"MatMulNBits with {bits} bits");
        }

        if (ins.Length > 4 && ins[4] is not null)
        {
            throw new NotSupportedException("MatMulNBits with g_idx");
        }

        var B = ConstIn(ins, 1, "quantized weight");
        var scales = ConstIn(ins, 2, "scales");
        var zps = OptConst(ins, 3, "zero points");
        long nb = (K + bs - 1) / bs;
        var wc = OnnxConst.Lazy(OnnxType.Float, [N, K], () =>
        {
            var q = B.Bytes();
            var sc = scales.Floats();
            var zb = zps?.Bytes();
            var res = new float[N * K];
            long blobPerBlock = bs * bits / 8;
            long zpRow = (nb * bits + 7) / 8;
            for (long r = 0; r < N; r++)
            {
                for (long k = 0; k < K; k++)
                {
                    long blk = k / bs;
                    long j = k % bs;
                    int v = bits == 4 ? (q[(r * nb + blk) * blobPerBlock + j / 2] >> (int)(4 * (j & 1))) & 0xF : q[(r * nb + blk) * blobPerBlock + j];
                    int z = bits == 4 ? 8 : 128;
                    if (zb is not null)
                    {
                        z = bits == 4 ? (zb[r * zpRow + blk / 2] >> (int)(4 * (blk & 1))) & 0xF : zb[r * nb + blk];
                    }

                    res[r * K + k] = (float)((v - z) * sc[r * nb + blk]);
                }
            }

            var bytes = new byte[res.Length * 4];
            Buffer.BlockCopy(res, 0, bytes, 0, bytes.Length);
            return bytes;
        }, Nm(n) + ".dequant");
        var y = _b.MatMul(X(ins, 0), WeightValue(wc, false, Nm(n) + ".weight"));
        Heavy();
        if (ins.Length > 5 && ins[5] is not null)
        {
            y = _b.Add(y, X(ins, 5));
        }

        SetDyn(n, 0, y);
    }
}
