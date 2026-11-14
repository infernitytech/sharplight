using System.Buffers.Binary;
using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Onnx;

// com.microsoft GroupQueryAttention and RotaryEmbedding (onnxruntime-genai / Optimum decoder exports).
internal sealed partial class OnnxTranslator
{
    /// <summary>
    /// GroupQueryAttention: query [B, S, H·D] (or packed QKV [B, S, (H + 2·Hkv)·D]), key/value [B, S, Hkv·D] and past key/value
    /// [B, Hkv, P, D] → output [B, S, H·D] and present key/value [B, Hkv, max(P, total), D]. Row b holds seqlens_k[b] + 1 tokens after this
    /// step, so its new tokens start at p_b = seqlens_k[b] + 1 - S (0 for a first prompt, S = total_sequence_length) and are written into
    /// the past buffer there; a constant uniform p_b is resolved at import, a runtime one by a one-hot scatter. Rotary embedding
    /// (do_rotary, at p_b + i or position_ids), per-head Q/K RMS norm, causal and local-window masking, softcap, smooth softmax, head
    /// sinks, an additive attention bias and the qk output are supported; quantized KV caches are not.
    /// </summary>
    private void GroupQueryAttention(OnnxNode n, TranslatedTensor?[] ins)
    {
        int heads = (int)n.Int("num_heads", 0);
        int kvHeads = (int)n.Int("kv_num_heads", 0);
        string name = Nm(n);
        if (Opt(ins, 12) is not null || Opt(ins, 13) is not null || n.Int("kv_cache_bit_width", 0) != 0 || n.Int("sliding_window_cache", 0) != 0)
        {
            throw new NotSupportedException("GroupQueryAttention with a quantized or window-sized KV cache");
        }

        var query = X(ins, 0);
        long s = query.Shape[1].Fixed;
        Value q, k, v;
        long d;
        if (Opt(ins, 1) is null)
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

        long past = Opt(ins, 3) is null ? 0 : X(ins, 3).Shape[2].Fixed;
        // The present length max(P, total) is a static shape, so a past buffer needs a constant total length (a folded Shape in exports).
        long total = past == 0 ? s : ConstIn(ins, 6, "GroupQueryAttention total_sequence_length").ScalarInt();
        long t = Math.Max(past, total);
        bool firstPrompt = total == s;

        // Start positions of the new tokens: constant (uniform or per row) or a runtime [B] value.
        long[]? starts = null;
        Value? start = null;
        if (firstPrompt)
        {
            starts = [0];
        }
        else if (ins[5] is { IsConst: true } lengths)
        {
            starts = [.. lengths.C!.Ints().Select(l => l + 1 - s)];
        }
        else
        {
            start = _b.Scale(AsValue(ins[5]!), 1f, 1 - s);
        }

        if (starts is not null && starts.Distinct().Count() == 1)
        {
            starts = [starts[0]];
        }
        else if (starts is not null)
        {
            start = F32Const([.. starts.Select(p => (float)p)], name + ".starts");
            starts = null;
        }

        // [B, S, h·D] → [B, S, h, D]; per-head RMS norm and rotary before the heads move outward.
        var qh = Rs(q, 0, s, heads, d);
        var kh = Rs(k, 0, s, kvHeads, d);
        if (Opt(ins, 14) is not null)
        {
            float eps = n.Float("qk_norm_epsilon", 1e-6f);
            qh = _b.RmsNorm(qh, X(ins, 14), eps);
            kh = _b.RmsNorm(kh, X(ins, 15), eps);
        }

        if (n.Int("do_rotary", 0) != 0)
        {
            var cosCache = ConstIn(ins, 7, "GroupQueryAttention cos_cache");
            var sinCache = ConstIn(ins, 8, "GroupQueryAttention sin_cache");
            Value cos, sin;
            if (Opt(ins, 9) is { } positionIds)
            {
                (cos, sin) = RuntimeRotaryRows(cosCache, sinCache, Idx(positionIds), s, name);
            }
            else if (starts is [var p])
            {
                (cos, sin) = RotaryTables(cosCache, sinCache, [.. Enumerable.Range(0, (int)s).Select(i => p + i)], name);
            }
            else
            {
                var positions = _b.Add(Rs(start!, 0, 1), F32Const([.. Enumerable.Range(0, (int)s).Select(i => (float)i)], name + ".offsets"));
                (cos, sin) = RuntimeRotaryRows(cosCache, sinCache, Emit("Cast", [positions], Attributes.From(("type", "i32"))), s, name);
            }

            bool interleaved = n.Int("rotary_interleaved", 0) != 0;
            qh = Rotate(qh, cos, sin, interleaved);
            kh = Rotate(kh, cos, sin, interleaved);
        }

        qh = _b.Permute(qh, 0, 2, 1, 3);
        kh = _b.Permute(kh, 0, 2, 1, 3);
        var vh = _b.Permute(Rs(v, 0, s, kvHeads, d), 0, 2, 1, 3);
        if (Opt(ins, 3) is not null)
        {
            kh = WriteCache(X(ins, 3), kh, t, starts, start, name + ".k");
            vh = WriteCache(X(ins, 4), vh, t, starts, start, name + ".v");
        }
        else if (t > s)
        {
            throw new NotSupportedException($"GroupQueryAttention without a past buffer and total length {t} > {s}");
        }

        if (n.Outputs.Count > 1 && n.Outputs[1] != "")
        {
            SetDyn(n, 1, kh);
        }

        if (n.Outputs.Count > 2 && n.Outputs[2] != "")
        {
            SetDyn(n, 2, vh);
        }

        var bias = GqaMask(s, t, starts, start, n.Int("causal", 1) != 0, n.Int("local_window_size", -1), name);
        if (Opt(ins, 10) is not null)
        {
            bias = _b.Add(bias, X(ins, 10));
        }

        Value? sink = null;
        if (Opt(ins, 11) is not null)
        {
            sink = Rs(X(ins, 11), 1, heads, 1, 1);
        }
        else if (n.Int("smooth_softmax", 0) != 0)
        {
            sink = F32Const([0f], name + ".sink");
        }

        float scale = n.Float("scale", 0f) is var sc && sc != 0 ? sc : 1f / MathF.Sqrt(d);
        var (y, scores, probabilities) = MultiHeadCore(qh, kh, vh, scale, bias, n.Float("softcap", 0f), sink, name);
        SetDyn(n, 0, MergeHeads(y, name));
        if (n.Outputs.Count > 3 && n.Outputs[3] != "")
        {
            SetDyn(n, 3, n.Int("qk_output", 0) == 2 ? probabilities : scores);
        }
    }

    /// <summary>
    /// The additive GroupQueryAttention mask: key j is visible to query i of row b iff j &lt;= p_b + i (causal; j &lt; p_b + S otherwise) and
    /// j &gt; p_b + i - window (onnxruntime's local window counts the query's own key). A constant [S, T] for a constant start, else [B, 1, S, T] from the runtime starts [B].
    /// </summary>
    private Value GqaMask(long s, long t, long[]? starts, Value? start, bool causal, long window, string name)
    {
        if (starts is [var p0])
        {
            var visible = new float[s * t];
            for (long i = 0; i < s; i++)
            {
                for (long j = 0; j < t; j++)
                {
                    long position = p0 + i;
                    bool sees = (causal ? j <= position : j < p0 + s) && (window < 0 || j > position - window);
                    visible[i * t + j] = sees ? 0f : float.MinValue;
                }
            }

            return _b.Constant(name + ".mask", DataType.F32, new Shape(s, t), ToBytes(visible));
        }

        float[] Grid(Func<long, long, float> f)
        {
            var g = new float[s * t];
            for (long i = 0; i < s; i++)
            {
                for (long j = 0; j < t; j++)
                {
                    g[i * t + j] = f(i, j);
                }
            }

            return g;
        }

        var p = Rs(start!, 0, 1, 1, 1);
        var allowed = _b.Binary("LessEqual",
            _b.Constant(name + ".frontier", DataType.F32, new Shape(s, t), ToBytes(Grid((i, j) => causal ? j - i : j - s + 1))), p);
        if (window >= 0)
        {
            // j - i > p_b - window
            var diagonal = _b.Constant(name + ".diagonal", DataType.F32, new Shape(s, t), ToBytes(Grid((i, j) => j - i)));
            allowed = _b.Mul(allowed, _b.Binary("Greater", diagonal, _b.Scale(p, 1f, -window)));
        }

        // (1 - allowed) · MinValue
        return _b.Scale(allowed, -float.MinValue, float.MinValue);
    }

    /// <summary>
    /// The present buffer [B, H, T, D]: the past rows before the start row of the new keys or values [B, H, S, D], then those, then zeros
    /// (as onnxruntime writes it). The start is the constant <paramref name="starts"/>[0], or the runtime rows <paramref name="start"/> [B]
    /// placed through a one-hot selection [B, 1, T, S].
    /// </summary>
    private Value WriteCache(Value pastBuffer, Value fresh, long t, long[]? starts, Value? start, string name)
    {
        long past = pastBuffer.Shape[2].Fixed;
        long s = fresh.Shape[2].Fixed;
        Value Grow(Value x, long rows)
        {
            var pads = new long[8];
            pads[6] = t - rows;
            return rows == t ? x : Emit("Pad", [x], Attributes.From(("pads", pads), ("mode", "constant"), ("value", 0f)));
        }

        if (starts is [var p])
        {
            if (p < 0 || p + s > t)
            {
                throw new InvalidDataException($"GroupQueryAttention writes rows {p}..{p + s} of a {t}-row cache");
            }

            var written = p > 0 ? Emit("Concat", [_b.Slice(pastBuffer, 2, 0, Math.Min(p, past)), fresh], Attributes.From(("axis", 2L))) : fresh;
            return Grow(written, written.Shape[2].Fixed);
        }

        var offsets = new float[t * s];
        for (long j = 0; j < t; j++)
        {
            for (long i = 0; i < s; i++)
            {
                offsets[j * s + i] = j - i;
            }
        }

        var first = Rs(start!, 0, 1, 1, 1);
        var select = _b.Binary("Equal", _b.Constant(name + ".rows", DataType.F32, new Shape(t, s), ToBytes(offsets)), first);
        var rows = _b.Constant(name + ".row", DataType.F32, new Shape(t, 1), ToBytes([.. Enumerable.Range(0, (int)t).Select(j => (float)j)]));
        var kept = _b.Mul(Grow(pastBuffer, past), _b.Binary("Less", rows, first));
        return _b.Add(kept, Emit("BatchMatMul", [select, fresh]));
    }

    /// <summary>cos/sin cache rows of runtime positions [B, S] (i32) as [B, S, 1, R/2].</summary>
    private (Value Cos, Value Sin) RuntimeRotaryRows(OnnxConst cosCache, OnnxConst sinCache, Value positions, long s, string name)
    {
        long half = cosCache.Dims[1];
        Value Rows(OnnxConst cache, string suffix)
            => Rs(Emit("GatherAxis", [ConstValue(cache, name + suffix), positions], Attributes.From(("axis", 0L))), 0, s, 1, half);
        return (Rows(cosCache, ".cos_cache"), Rows(sinCache, ".sin_cache"));
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
            // Runtime position ids [B, S], or a runtime offset [1] of positions offset + 0..S-1: gather the cache rows → [B or 1, S, 1, R/2].
            var ids = Idx(ins[1]!);
            bool offset = ids.Rank == 1 && ids.Shape[0].Fixed == 1;
            if (ids.Rank != 2 && !offset)
            {
                throw new NotSupportedException("RotaryEmbedding with runtime position ids that are not [B, S] or a start offset");
            }

            if (offset)
            {
                var positions = _b.Add(AsValue(ins[1]!), F32Const([.. Enumerable.Range(0, (int)s).Select(i => (float)i)], Nm(n) + ".offsets"));
                ids = Emit("Cast", [positions], Attributes.From(("type", "i32")));
            }

            long half = ConstIn(ins, 2, "RotaryEmbedding cache").Dims[1];
            Value Rows(int input)
            {
                var rows = Emit("GatherAxis", [X(ins, input), ids], Attributes.From(("axis", 0L)));
                return offset ? Rs(rows, s, 1, half) : Rs(rows, 0, s, 1, half);
            }

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
