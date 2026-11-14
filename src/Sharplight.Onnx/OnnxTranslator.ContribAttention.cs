using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Onnx;

// com.microsoft Attention and MultiHeadAttention (onnxruntime transformer optimizer exports) on a shared multi-head core.
internal sealed partial class OnnxTranslator
{
    /// <summary>
    /// Scaled dot-product attention over heads: q [B, Hq, S, D], k [B, Hkv, T, D], v [B, Hkv, T, Dv] → [B, Hq, S, Dv]. The query heads
    /// of a group are adjacent (head h reads key/value head h / (Hq / Hkv)), so grouped heads share K/V by folding a group into rows.
    /// <paramref name="bias"/> is additive and broadcastable to [B, Hq, S, T]; <paramref name="sink"/> (logits broadcastable to
    /// [1, Hq, 1, 1]) adds a virtual key with zero value to every softmax row. Also returns the scaled scores (before softcap and bias)
    /// and the probabilities.
    /// </summary>
    private (Value Output, Value Scores, Value Probabilities) MultiHeadCore(Value q, Value k, Value v, float scale, Value? bias, float softcap, Value? sink, string name)
    {
        long hq = q.Shape[1].Fixed;
        long s = q.Shape[2].Fixed;
        long d = q.Shape[3].Fixed;
        long hkv = k.Shape[1].Fixed;
        long t = k.Shape[2].Fixed;
        long dv = v.Shape[3].Fixed;
        if (hq % hkv != 0)
        {
            throw new NotSupportedException($"attention with {hq} query heads over {hkv} key/value heads");
        }

        long reps = hq / hkv;
        var grouped = reps > 1 ? Rs(q, 0, hkv, reps * s, d) : q;
        var scores = Emit("BatchMatMul", [grouped, k], Attributes.From(("trans_b", true), ("alpha", scale)));
        Heavy();
        if (reps > 1)
        {
            scores = Rs(scores, 0, hq, s, t);
        }

        var logits = scores;
        if (softcap > 0)
        {
            logits = _b.Scale(_b.Unary("Tanh", _b.Scale(logits, 1f / softcap)), softcap);
        }

        if (bias is not null)
        {
            logits = _b.Add(logits, bias);
        }

        Value probabilities;
        if (sink is null)
        {
            probabilities = _b.Emit1("Softmax", [logits]);
        }
        else
        {
            // One more column holding the sink logit; its probability is dropped (the sink's value is zero).
            var column = _b.Add(_b.Scale(_b.Slice(scores, 3, 0, 1), 0f), sink);
            var extended = _b.Emit1("Softmax", [Emit("Concat", [logits, column], Attributes.From(("axis", 3L)))]);
            probabilities = _b.Slice(extended, 3, 0, t);
        }

        var weights = reps > 1 ? Rs(probabilities, 0, hkv, reps * s, t) : probabilities;
        var y = Emit("BatchMatMul", [weights, v], null, name + ".heads");
        Heavy();
        return (reps > 1 ? Rs(y, 0, hq, s, dv) : y, scores, probabilities);
    }

    /// <summary>[B, H, S, D] → [B, S, H·D].</summary>
    private Value MergeHeads(Value x, string? name = null)
        => Emit("Reshape", [Permute(x, [0, 2, 1, 3])], Attributes.From(("shape", L(0, x.Shape[2].Fixed, x.Shape[1].Fixed * x.Shape[3].Fixed))), name);

    /// <summary>
    /// The additive mask of the contrib attention operators, broadcastable to [B, H, S, T] (null when nothing is masked): masked scores get
    /// <paramref name="filter"/> (mask_filter_value) added, as onnxruntime does. <paramref name="mask"/> is a mask_index / key_padding_mask:
    /// key lengths [B], end and start positions [2·B], a raw 0/1 mask [B, T] or [B, S, T]; <paramref name="causal"/> hides key j from
    /// query i when j &gt; i + <paramref name="past"/>.
    /// </summary>
    private Value? ContribMaskBias(TranslatedTensor? mask, Dim batch, long s, long t, long past, bool causal, float filter, string name)
    {
        Value? allowed = null;
        if (mask is not null)
        {
            var m = AsValue(mask);
            var keys = F32Const([.. Enumerable.Range(0, (int)t).Select(j => (float)j)], name + ".keys");
            switch (m.Rank)
            {
                case 1 when m.Shape[0] == batch:
                    allowed = _b.Binary("Less", keys, Rs(m, 0, 1, 1, 1));
                    break;
                case 1 when !batch.IsSymbolic && m.Shape[0].Fixed == 2 * batch.Fixed:
                {
                    long b = batch.Fixed;
                    var end = Rs(_b.Slice(m, 0, 0, b), b, 1, 1, 1);
                    var start = Rs(_b.Slice(m, 0, b, b), b, 1, 1, 1);
                    allowed = _b.Mul(_b.Binary("Less", keys, end), _b.Binary("GreaterEqual", keys, start));
                    break;
                }

                case 2:
                    allowed = Rs(PadKeys(m, t), 0, 1, 1, t);
                    break;
                case 3:
                    allowed = Rs(PadKeys(m, t), 0, 1, m.Shape[1].Fixed, t);
                    break;
                default:
                    throw new NotSupportedException($"attention mask of shape {m.Shape}");
            }
        }

        if (causal)
        {
            var visible = new float[s * t];
            for (long i = 0; i < s; i++)
            {
                for (long j = 0; j < t; j++)
                {
                    visible[i * t + j] = j <= i + past ? 1f : 0f;
                }
            }

            var constant = _b.Constant(name + ".causal", DataType.F32, new Shape(s, t), ToBytes(visible));
            allowed = allowed is null ? constant : _b.Mul(allowed, constant);
        }

        // (1 - allowed) · filter
        return allowed is null ? null : _b.Scale(allowed, -filter, filter);
    }

    /// <summary>A raw key mask narrower than the keys (past keys not covered) is padded with visible keys at the front.</summary>
    private Value PadKeys(Value mask, long t)
    {
        long width = mask.Shape[mask.Rank - 1].Fixed;
        if (width == t)
        {
            return mask;
        }

        if (width > t)
        {
            throw new InvalidDataException($"attention mask covers {width} keys, more than the {t} keys");
        }

        var pads = new long[2 * mask.Rank];
        pads[mask.Rank - 1] = t - width;
        return Emit("Pad", [mask], Attributes.From(("pads", pads), ("mode", "constant"), ("value", 1f)));
    }

    /// <summary>
    /// com.microsoft Attention: input [B, S, In] · weights [In, Dq + Dk + Dv] (+ bias), qkv_hidden_sizes, mask_index (key lengths, start and
    /// end positions, raw 2-D or 3-D masks), past [2, B, H, P, D] → present [2, B, H, P + S, D], unidirectional, attention_bias and scale.
    /// </summary>
    private void ContribAttention(OnnxNode n, TranslatedTensor?[] ins)
    {
        if (Opt(ins, 6) is not null || n.Int("past_present_share_buffer", 0) != 0 || n.Int("do_rotary", 0) != 0)
        {
            throw new NotSupportedException("Attention with a shared past/present buffer or do_rotary");
        }

        string name = Nm(n);
        var x = X(ins, 0);
        Value qkv;
        if (ins[1]!.IsConst)
        {
            qkv = _b.MatMul(x, WeightValue(ins[1]!.C!.Transposed([1, 0]), true, name + ".qkv"));
        }
        else
        {
            qkv = Emit("BatchMatMul", [x, X(ins, 1)]);
        }

        Heavy();
        if (Opt(ins, 2) is not null)
        {
            qkv = _b.Add(qkv, X(ins, 2));
        }

        long width = qkv.Shape[2].Fixed;
        long[] sizes = n.Ints("qkv_hidden_sizes") is { Length: 3 } given ? given : [width / 3, width / 3, width / 3];
        int heads = (int)n.Int("num_heads", 1);
        long s = x.Shape[1].Fixed;
        var q = SplitHeads(_b.Slice(qkv, 2, 0, sizes[0]), heads);
        var k = SplitHeads(_b.Slice(qkv, 2, sizes[0], sizes[1]), heads);
        var v = SplitHeads(_b.Slice(qkv, 2, sizes[0] + sizes[1], sizes[2]), heads);
        long past = 0;
        if (Opt(ins, 4) is not null)
        {
            var pastKv = X(ins, 4);
            past = pastKv.Shape[3].Fixed;
            long b = pastKv.Shape[1].Fixed;
            k = Emit("Concat", [Rs(_b.Slice(pastKv, 0, 0, 1), b, heads, past, sizes[0] / heads), k], Attributes.From(("axis", 2L)));
            v = Emit("Concat", [Rs(_b.Slice(pastKv, 0, 1, 1), b, heads, past, sizes[2] / heads), v], Attributes.From(("axis", 2L)));
        }

        if (n.Outputs.Count > 1 && n.Outputs[1] != "")
        {
            long b = k.Shape[0].Fixed;
            long t = k.Shape[2].Fixed;
            SetDyn(n, 1, Emit("Concat", [Rs(k, 1, b, heads, t, sizes[1] / heads), Rs(v, 1, b, heads, t, sizes[2] / heads)], Attributes.From(("axis", 0L))));
        }

        float scale = n.Float("scale", 0f) is var sc && sc != 0 ? sc : 1f / MathF.Sqrt(sizes[0] / heads);
        var bias = ContribMaskBias(Opt(ins, 3), x.Shape[0], s, k.Shape[2].Fixed, past, n.Int("unidirectional", 0) != 0, n.Float("mask_filter_value", -10000f), name);
        if (Opt(ins, 5) is not null)
        {
            bias = bias is null ? X(ins, 5) : _b.Add(bias, X(ins, 5));
        }

        var (y, _, _) = MultiHeadCore(q, k, v, scale, bias, 0f, null, name);
        SetDyn(n, 0, MergeHeads(y, name));
    }

    /// <summary>
    /// com.microsoft MultiHeadAttention: query [B, S, H·D] with key/value [B, L, H·D] (bias for all three), key/value already in heads
    /// [B, H, L, D] (cross attention), packed QKV [B, S, H, 3, D] or packed KV [B, L, H, 2, D]; key_padding_mask (key lengths or a raw mask),
    /// attention_bias, past key/value [B, H, P, D] → present key/value, unidirectional and scale.
    /// </summary>
    private void MultiHeadAttention(OnnxNode n, TranslatedTensor?[] ins)
    {
        if (Opt(ins, 8) is not null || Opt(ins, 9) is not null)
        {
            throw new NotSupportedException("MultiHeadAttention with past_sequence_length or cache_indirection");
        }

        string name = Nm(n);
        int heads = (int)n.Int("num_heads", 1);
        var query = X(ins, 0);
        long s = query.Shape[1].Fixed;
        Value q, k, v;
        bool kvInHeads = false;
        if (query.Rank == 5)
        {
            // Packed QKV [B, S, H, 3, D].
            long d = query.Shape[4].Fixed;
            Value Part(int i) => Rs(_b.Slice(query, 3, i, 1), 0, s, heads * d);
            (q, k, v) = (Part(0), Part(1), Part(2));
        }
        else
        {
            q = query;
            var key = X(ins, 1);
            if (key.Rank == 5)
            {
                long l = key.Shape[1].Fixed;
                long d = key.Shape[4].Fixed;
                k = Rs(_b.Slice(key, 3, 0, 1), 0, l, heads * d);
                v = Rs(_b.Slice(key, 3, 1, 1), 0, l, heads * d);
            }
            else
            {
                k = key;
                v = X(ins, 2);
                kvInHeads = key.Rank == 4;
            }
        }

        long hidden = q.Shape[2].Fixed;
        if (Opt(ins, 3) is not null)
        {
            var bias = X(ins, 3);
            long kHidden = kvInHeads ? k.Shape[1].Fixed * k.Shape[3].Fixed : k.Shape[2].Fixed;
            q = _b.Add(q, _b.Slice(bias, 0, 0, hidden));
            if (!kvInHeads)
            {
                k = _b.Add(k, _b.Slice(bias, 0, hidden, kHidden));
                v = _b.Add(v, _b.Slice(bias, 0, hidden + kHidden, v.Shape[2].Fixed));
            }
        }

        var qh = SplitHeads(q, heads);
        var kh = kvInHeads ? k : SplitHeads(k, heads);
        var vh = kvInHeads ? v : SplitHeads(v, heads);
        long past = 0;
        if (Opt(ins, 6) is not null && !kvInHeads)
        {
            var pastKey = X(ins, 6);
            past = pastKey.Shape[2].Fixed;
            kh = Emit("Concat", [pastKey, kh], Attributes.From(("axis", 2L)));
            vh = Emit("Concat", [X(ins, 7), vh], Attributes.From(("axis", 2L)));
        }

        if (n.Outputs.Count > 1 && n.Outputs[1] != "")
        {
            SetDyn(n, 1, kh);
        }

        if (n.Outputs.Count > 2 && n.Outputs[2] != "")
        {
            SetDyn(n, 2, vh);
        }

        float scale = n.Float("scale", 0f) is var sc && sc != 0 ? sc : 1f / MathF.Sqrt(qh.Shape[3].Fixed);
        var maskBias = ContribMaskBias(Opt(ins, 4), query.Shape[0], s, kh.Shape[2].Fixed, past, n.Int("unidirectional", 0) != 0,
            n.Float("mask_filter_value", -10000f), name);
        if (Opt(ins, 5) is not null)
        {
            maskBias = maskBias is null ? X(ins, 5) : _b.Add(maskBias, X(ins, 5));
        }

        var (y, scores, _) = MultiHeadCore(qh, kh, vh, scale, maskBias, 0f, null, name);
        SetDyn(n, 0, MergeHeads(y, name));
        if (n.Outputs.Count > 3 && n.Outputs[3] != "")
        {
            SetDyn(n, 3, scores);
        }
    }
}
