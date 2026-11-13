using System.Buffers.Binary;
using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Onnx;

// com.microsoft GroupQueryAttention and RotaryEmbedding (onnxruntime-genai / Optimum decoder exports).
internal sealed partial class OnnxTranslator
{
    /// <summary>
    /// GroupQueryAttention: query [B, S, H·D] (or packed QKV), key/value [B, S, Hkv·D], past key/value [B, Hkv, P, D] → output [B, S, H·D] and
    /// present key/value [B, Hkv, P+S, D]. Rotary embedding (do_rotary), causal and local-window masking, softcap and an additive attention
    /// bias are supported. Positions are taken as <c>total_sequence_length - S + i</c> for every batch row: rows of a batch must have the
    /// same length (padding is expressed through the attention bias).
    /// </summary>
    private void GroupQueryAttention(OnnxNode n, TranslatedTensor?[] ins)
    {
        int heads = (int)n.Int("num_heads", 0);
        int kvHeads = (int)n.Int("kv_num_heads", 0);
        if (n.Int("smooth_softmax", 0) != 0 || (ins.Length > 9 && ins[9] is not null) || (ins.Length > 11 && ins.Skip(11).Any(t => t is not null)))
        {
            throw new NotSupportedException("GroupQueryAttention with smooth_softmax, position_ids or head_sink");
        }

        var query = X(ins, 0);
        long s = query.Shape[1].Fixed;
        Value q, k, v;
        long d;
        if (ins[1] is null)
        {
            d = query.Shape[2].Fixed / (heads + 2 * kvHeads);
            q = _b.Slice(query, 2, 0, heads * d);
            k = _b.Slice(query, 2, heads * d, kvHeads * d);
            v = _b.Slice(query, 2, (heads + kvHeads) * d, kvHeads * d);
        }
        else
        {
            q = query;
            k = X(ins, 1);
            v = X(ins, 2);
            d = q.Shape[2].Fixed / heads;
        }

        long total = ConstIn(ins, 6, "GroupQueryAttention total_sequence_length").ScalarInt();
        long past = ins[3] is null ? 0 : X(ins, 3).Shape[2].Fixed;
        if (past + s != total)
        {
            throw new NotSupportedException($"GroupQueryAttention with past length {past}, {s} new tokens and total length {total} (a shared KV buffer)");
        }

        // [B, S, h·D] → [B, S, h, D], rotary, → [B, h, S, D]
        var qh = Rs(q, 0, s, heads, d);
        var kh = Rs(k, 0, s, kvHeads, d);
        if (n.Int("do_rotary", 0) != 0)
        {
            var (cos, sin) = RotaryTables(ConstIn(ins, 7, "GroupQueryAttention cos_cache"), ConstIn(ins, 8, "GroupQueryAttention sin_cache"),
                Enumerable.Range(0, (int)s).Select(i => total - s + i).ToArray(), Nm(n));
            bool interleaved = n.Int("rotary_interleaved", 0) != 0;
            qh = Rotate(qh, cos, sin, interleaved);
            kh = Rotate(kh, cos, sin, interleaved);
        }

        qh = _b.Permute(qh, 0, 2, 1, 3);
        kh = _b.Permute(kh, 0, 2, 1, 3);
        var vh = _b.Permute(Rs(v, 0, s, kvHeads, d), 0, 2, 1, 3);
        if (ins[3] is not null)
        {
            kh = Emit("Concat", [X(ins, 3), kh], Attributes.From(("axis", 2L)));
            vh = Emit("Concat", [X(ins, 4), vh], Attributes.From(("axis", 2L)));
        }

        if (n.Outputs.Count > 1 && n.Outputs[1] != "")
        {
            SetDyn(n, 1, kh);
        }

        if (n.Outputs.Count > 2 && n.Outputs[2] != "")
        {
            SetDyn(n, 2, vh);
        }

        // Each KV head serves heads/kvHeads query heads.
        if (heads != kvHeads)
        {
            var group = Enumerable.Range(0, heads).Select(h => h / (heads / kvHeads)).ToArray();
            var index = IndexConstant(Nm(n) + ".kvgroup", group);
            kh = Emit("GatherAxis", [kh, index], Attributes.From(("axis", 1L)));
            vh = Emit("GatherAxis", [vh, index], Attributes.From(("axis", 1L)));
        }

        float scale = n.Float("scale", 0f) is var sc && sc != 0 ? sc : 1f / MathF.Sqrt(d);
        var scores = Emit("BatchMatMul", [qh, kh], Attributes.From(("trans_b", true), ("alpha", scale)));
        Heavy();
        float softcap = n.Float("softcap", 0f);
        if (softcap > 0)
        {
            scores = _b.Scale(_b.Unary("Tanh", _b.Scale(scores, 1f / softcap)), softcap);
        }

        scores = _b.Add(scores, CausalMask(s, total, n.Int("local_window_size", -1), Nm(n)));
        if (ins.Length > 10 && ins[10] is not null)
        {
            scores = _b.Add(scores, X(ins, 10));
        }

        var probabilities = _b.Emit1("Softmax", [scores]);
        var o = Emit("BatchMatMul", [probabilities, vh]);
        Heavy();
        SetDyn(n, 0, Emit("Reshape", [_b.Permute(o, 0, 2, 1, 3)], Attributes.From(("shape", L(0, s, heads * d))), Nm(n)));
    }

    /// <summary>
    /// RotaryEmbedding: input [B, S, H·D] (with num_heads) or [B, H, S, D], position_ids [B, S] (constant) or a scalar offset, cos/sin caches
    /// [max, R/2]; rotates the first R = rotary_embedding_dim (default the cache width · 2) channels of every head.
    /// </summary>
    private void RotaryEmbedding(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        if (n.Int("is_packed_batching", 0) != 0)
        {
            throw new NotSupportedException("RotaryEmbedding with packed batching");
        }

        bool bnsh = x.Rank == 4;
        long s = x.Shape[bnsh ? 2 : 1].Fixed;
        Value cos, sin;
        if (ins[1]!.IsConst)
        {
            var positions = ins[1]!.C!;
            long[] pos;
            if (positions.Count == 1)
            {
                long offset = positions.Ints()[0];
                pos = Enumerable.Range(0, (int)s).Select(i => offset + i).ToArray();
            }
            else
            {
                var all = positions.Ints();
                pos = all.Take((int)s).ToArray();
                if (Enumerable.Range(1, (int)(all.Length / s) - 1).Any(b => !all.Skip(b * (int)s).Take((int)s).SequenceEqual(pos)))
                {
                    throw new NotSupportedException("RotaryEmbedding with different constant positions per batch row");
                }
            }

            (cos, sin) = RotaryTables(ConstIn(ins, 2, "RotaryEmbedding cos_cache"), ConstIn(ins, 3, "RotaryEmbedding sin_cache"), pos, Nm(n));
        }
        else
        {
            // Runtime position ids [B, S]: gather the cache rows → [B, S, 1, R/2].
            var ids = Idx(ins[1]!);
            if (ids.Rank != 2)
            {
                throw new NotSupportedException("RotaryEmbedding with runtime position ids that are not [B, S]");
            }

            Value Rows(int input) => Rs(Emit("GatherAxis", [X(ins, input), ids], Attributes.From(("axis", 0L))), 0, s, 1, ConstIn(ins, input, "RotaryEmbedding cache").Dims[1]);
            cos = Rows(2);
            sin = Rows(3);
        }

        int heads = (int)n.Int("num_heads", 0);
        Value xh;
        if (bnsh)
        {
            xh = _b.Permute(x, 0, 2, 1, 3);
        }
        else
        {
            long width = x.Shape[2].Fixed;
            long d = heads > 0 ? width / heads : 2 * cos.Shape[cos.Rank - 1].Fixed;
            xh = Rs(x, 0, s, width / d, d);
        }

        var y = Rotate(xh, cos, sin, n.Int("interleaved", 0) != 0);
        SetDyn(n, 0, bnsh ? _b.Permute(y, 0, 2, 1, 3) : Emit("Reshape", [y], Attributes.From(("shape", L(0, s, x.Shape[2].Fixed))), Nm(n)));
    }

    /// <summary>cos/sin rows of the given positions as constants [S, 1, R/2] (broadcast over heads).</summary>
    private (Value Cos, Value Sin) RotaryTables(OnnxConst cosCache, OnnxConst sinCache, long[] positions, string name)
    {
        long half = cosCache.Dims[1];
        Value Rows(OnnxConst cache, string suffix)
        {
            var src = cache.Floats();
            var bytes = new byte[positions.Length * half * 4];
            for (int i = 0; i < positions.Length; i++)
            {
                if (positions[i] < 0 || positions[i] >= cache.Dims[0])
                {
                    throw new NotSupportedException($"rotary position {positions[i]} outside the cache");
                }

                for (long j = 0; j < half; j++)
                {
                    BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan((int)((i * half + j) * 4), 4), (float)src[positions[i] * half + j]);
                }
            }

            return _b.Constant(name + suffix, DataType.F32, new Shape(positions.Length, 1, half), bytes);
        }

        return (Rows(cosCache, ".cos"), Rows(sinCache, ".sin"));
    }

    /// <summary>Rotary embedding of x [B, S, H, D] (rotating the first 2·half channels; interleaved pairs or two halves).</summary>
    private Value Rotate(Value x, Value cos, Value sin, bool interleaved)
    {
        long d = x.Shape[3].Fixed;
        long half = cos.Shape[cos.Rank - 1].Fixed;
        long s = x.Shape[1].Fixed;
        long h = x.Shape[2].Fixed;
        var rotated = 2 * half == d ? x : _b.Slice(x, 3, 0, 2 * half);
        Value x1, x2;
        if (interleaved)
        {
            var pairs = Rs(rotated, 0, s, h, half, 2);
            x1 = Rs(_b.Slice(pairs, 4, 0, 1), 0, s, h, half);
            x2 = Rs(_b.Slice(pairs, 4, 1, 1), 0, s, h, half);
        }
        else
        {
            x1 = _b.Slice(rotated, 3, 0, half);
            x2 = _b.Slice(rotated, 3, half, half);
        }

        var r1 = _b.Binary("Sub", _b.Binary("Mul", x1, cos), _b.Binary("Mul", x2, sin));
        var r2 = _b.Add(_b.Binary("Mul", x2, cos), _b.Binary("Mul", x1, sin));
        var y = interleaved
            ? Rs(Emit("Concat", [Rs(r1, 0, s, h, half, 1), Rs(r2, 0, s, h, half, 1)], Attributes.From(("axis", 4L))), 0, s, h, 2 * half)
            : Emit("Concat", [r1, r2], Attributes.From(("axis", 3L)));
        return 2 * half == d ? y : Emit("Concat", [y, _b.Slice(x, 3, 2 * half, d - 2 * half)], Attributes.From(("axis", 3L)));
    }

    /// <summary>Additive mask [S, T] for the last S of T positions: causal, and limited to a local window when window ≥ 0.</summary>
    private Value CausalMask(long s, long total, long window, string name)
    {
        var mask = new byte[s * total * 4];
        for (long i = 0; i < s; i++)
        {
            long position = total - s + i;
            for (long j = 0; j < total; j++)
            {
                bool visible = j <= position && (window < 0 || j >= position - window);
                BinaryPrimitives.WriteSingleLittleEndian(mask.AsSpan((int)((i * total + j) * 4), 4), visible ? 0f : float.MinValue);
            }
        }

        return _b.Constant(name + ".mask", DataType.F32, new Shape(s, total), mask);
    }

    private Value IndexConstant(string name, int[] values)
    {
        var bytes = new byte[values.Length * 4];
        for (int i = 0; i < values.Length; i++)
        {
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(i * 4), values[i]);
        }

        return _b.Constant(name, DataType.I32, new Shape(values.Length), bytes);
    }
}
