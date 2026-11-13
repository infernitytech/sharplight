using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Onnx;

// ai.onnx Attention (opset 23+): scaled dot-product attention with masks, KV cache and grouped heads.
internal sealed partial class OnnxTranslator
{
    // ONNX operators whose outputs are boolean tensors (used to tell bool masks from additive float masks: both travel as f32).
    private static readonly HashSet<string> s_boolProducers =
        ["Equal", "Less", "Greater", "LessOrEqual", "GreaterOrEqual", "And", "Or", "Xor", "Not", "IsNaN", "IsInf"];

    /// <summary>
    /// Attention: Q [B, Hq, S, D] / K [B, Hkv, L, D] / V [B, Hkv, L, Dv] (or 3-D [B, S, H·D] with q_num_heads / kv_num_heads), optional
    /// attn_mask (bool or additive, broadcast to [B, Hq, S, T], shorter key axes padded as masked), past key/value [B, Hkv, P, D]
    /// (concatenated into present key/value, outputs 1 and 2), nonpad_kv_seqlen [B] (external cache: valid keys per batch row),
    /// is_causal (bottom-right aligned), left/right window sizes, scale, softcap and the qk_matmul_output (output 3, mode 0–3).
    /// Translated as BatchMatMul + Softmax + BatchMatMul; grouped heads share K/V by folding the query heads of a group into its rows.
    /// A query row without any visible key yields zeros.
    /// </summary>
    private void OnnxAttention(OnnxNode n, TranslatedTensor?[] ins)
    {
        if (TryFusedAttention(n, ins))
        {
            return;
        }

        var q = X(ins, 0);
        var k = X(ins, 1);
        var v = X(ins, 2);
        bool threeD = q.Rank == 3;
        if (threeD)
        {
            long qHeads = n.Int("q_num_heads", 0);
            long kvHeads = n.Int("kv_num_heads", 0);
            if (qHeads <= 0 || kvHeads <= 0)
            {
                throw new InvalidDataException("3-D Attention needs q_num_heads and kv_num_heads");
            }

            q = SplitHeads(q, qHeads);
            k = SplitHeads(k, kvHeads);
            v = SplitHeads(v, kvHeads);
        }
        else if (q.Rank != 4)
        {
            throw new NotSupportedException($"Attention with rank-{q.Rank} inputs");
        }

        long hq = q.Shape[1].Fixed;
        long s = q.Shape[2].Fixed;
        long d = q.Shape[3].Fixed;
        long hkv = k.Shape[1].Fixed;
        long dv = v.Shape[3].Fixed;
        long past = 0;
        if (Opt(ins, 4) is not null)
        {
            var pastKey = X(ins, 4);
            past = pastKey.Shape[2].Fixed;
            k = Emit("Concat", [pastKey, k], Attributes.From(("axis", 2L)));
            v = Emit("Concat", [X(ins, 5), v], Attributes.From(("axis", 2L)));
        }

        if (n.Outputs.Count > 1 && n.Outputs[1] != "")
        {
            SetDyn(n, 1, k);
        }

        if (n.Outputs.Count > 2 && n.Outputs[2] != "")
        {
            SetDyn(n, 2, v);
        }

        long t = k.Shape[2].Fixed;
        if (hq % hkv != 0)
        {
            throw new NotSupportedException($"Attention with {hq} query heads over {hkv} key/value heads");
        }

        long reps = hq / hkv;
        float scale = n.Has("scale") ? n.Float("scale", 1f) : 1f / MathF.Sqrt(d);

        // The query heads of a group are adjacent, so [B, Hq, S, D] = [B, Hkv, reps·S, D] shares each K/V head without copying it.
        var grouped = reps > 1 ? Rs(q, 0, hkv, reps * s, d) : q;
        var scores = Emit("BatchMatMul", [grouped, k], Attributes.From(("trans_b", true), ("alpha", scale)));
        Heavy();
        if (reps > 1)
        {
            scores = Rs(scores, 0, hq, s, t);
        }

        float softcap = n.Float("softcap", 0f);
        if (softcap > 0)
        {
            scores = _b.Scale(_b.Unary("Tanh", _b.Scale(scores, 1f / softcap)), softcap);
        }

        var mask = AttentionBias(n, ins, s, t, past);
        var biased = mask.Bias is null ? scores : _b.Add(scores, mask.Bias);
        var probabilities = _b.Emit1("Softmax", [biased]);
        if (mask.Visible is not null)
        {
            probabilities = Emit("Where", [mask.Visible, probabilities, ConstValue(OnnxConst.Scalar(0f), "attn.zero")]);
        }

        if (n.Outputs.Count > 3 && n.Outputs[3] != "")
        {
            long mode = n.Int("qk_matmul_output_mode", 0);
            SetDyn(n, 3, mode switch
            {
                0 or 1 => scores,
                2 => biased,
                3 => probabilities,
                _ => throw new NotSupportedException($"Attention qk_matmul_output_mode {mode}"),
            });
        }

        var weights = reps > 1 ? Rs(probabilities, 0, hkv, reps * s, t) : probabilities;
        var y = Emit("BatchMatMul", [weights, v]);
        Heavy();
        if (reps > 1)
        {
            y = Rs(y, 0, hq, s, dv);
        }

        SetDyn(n, 0, threeD ? Emit("Reshape", [Permute(y, [0, 2, 1, 3])], Attributes.From(("shape", L(0, s, hq * dv))), Nm(n)) : y);
    }

    /// <summary>
    /// The plain case (batch 1, no mask, cache or window, causal only as self-attention, head sizes the kernels vectorize) maps onto the
    /// fused IR Attention op, which providers run as flash attention.
    /// </summary>
    private bool TryFusedAttention(OnnxNode n, TranslatedTensor?[] ins)
    {
        if (ins.Skip(3).Any(t => t is not null) || n.Outputs.Skip(1).Any(o => o != "") || n.Int("left_window_size", -1) >= 0 || n.Int("right_window_size", -1) >= 0)
        {
            return false;
        }

        var q = X(ins, 0);
        var k = X(ins, 1);
        var v = X(ins, 2);
        if (q.Shape[0].IsSymbolic || q.Shape[0].Fixed != 1 || q.Rank is not (3 or 4))
        {
            return false;
        }

        bool threeD = q.Rank == 3;
        long hq = threeD ? n.Int("q_num_heads", 0) : q.Shape[1].Fixed;
        long hkv = threeD ? n.Int("kv_num_heads", 0) : k.Shape[1].Fixed;
        if (hq <= 0 || hkv <= 0 || hq % hkv != 0)
        {
            return false;
        }

        // [1, S, H·D] → [S, H, D]; [1, H, S, D] → [S, H, D] (a view).
        Value Rows(Value x, long heads) => threeD
            ? Rs(x, x.Shape[1].Fixed, heads, x.Shape[2].Fixed / heads)
            : Permute(Rs(x, heads, x.Shape[2].Fixed, x.Shape[3].Fixed), [1, 0, 2]);
        var qh = Rows(q, hq);
        var kh = Rows(k, hkv);
        var vh = Rows(v, hkv);
        long s = qh.Shape[0].Fixed;
        long t = kh.Shape[0].Fixed;
        long d = qh.Shape[2].Fixed;
        long dv = vh.Shape[2].Fixed;
        bool causal = n.Int("is_causal", 0) != 0;
        if (d % 8 != 0 || dv % 8 != 0 || (causal && s != t))
        {
            return false;
        }

        float scale = n.Has("scale") ? n.Float("scale", 1f) : 1f / MathF.Sqrt(d);
        var attributes = Attributes.From(("scale", scale), ("causal", causal), ("softcap", Math.Max(0f, n.Float("softcap", 0f))));
        Value[] operands = [qh, kh, vh];
        if (causal)
        {
            // One span over all keys; "causal" limits row i to keys ≤ i.
            operands = [.. operands, Rs(IndexConstant(Nm(n) + ".spans", [.. Enumerable.Range(0, (int)s).SelectMany(_ => new[] { 0, (int)t })]), s, 2)];
        }

        var y = Emit("Attention", operands, attributes);
        Heavy();
        SetDyn(n, 0, threeD
            ? Emit("Reshape", [y], Attributes.From(("shape", L(1, s, hq * dv))), Nm(n))
            : Emit("Reshape", [Permute(y, [1, 0, 2])], Attributes.From(("shape", L(1, hq, s, dv))), Nm(n)));
        return true;
    }

    /// <summary>
    /// ai.onnx.preview FlexAttention: Q [B, Hq, L, E], K [B, Hkv, S, E], V [B, Hkv, S, Ev]; the optional score_mod and prob_mod graphs
    /// (one tensor in, one out) are inlined on the scaled scores [B, Hq, L, S] and on the softmax probabilities.
    /// </summary>
    private void FlexAttention(OnnxNode n, TranslatedTensor?[] ins)
    {
        var q = X(ins, 0);
        var k = X(ins, 1);
        var v = X(ins, 2);
        if (q.Rank != 4 || k.Rank != 4 || v.Rank != 4)
        {
            throw new InvalidDataException("FlexAttention needs rank-4 Q, K and V");
        }

        long hq = q.Shape[1].Fixed;
        long l = q.Shape[2].Fixed;
        long e = q.Shape[3].Fixed;
        long hkv = k.Shape[1].Fixed;
        long s = k.Shape[2].Fixed;
        long ev = v.Shape[3].Fixed;
        long reps = hq / hkv;
        float scale = n.Has("scale") ? n.Float("scale", 1f) : 1f / MathF.Sqrt(e);
        var grouped = reps > 1 ? Rs(q, 0, hkv, reps * l, e) : q;
        var scores = Emit("BatchMatMul", [grouped, k], Attributes.From(("trans_b", true), ("alpha", scale)));
        Heavy();
        scores = reps > 1 ? Rs(scores, 0, hq, l, s) : scores;
        scores = ApplyModifier(n, "score_mod", scores);
        var probabilities = ApplyModifier(n, "prob_mod", _b.Emit1("Softmax", [scores]));
        var y = Emit("BatchMatMul", [reps > 1 ? Rs(probabilities, 0, hkv, reps * l, s) : probabilities, v]);
        Heavy();
        SetDyn(n, 0, reps > 1 ? Rs(y, 0, hq, l, ev) : y);
    }

    /// <summary>Inlines a one-input, one-output graph attribute on <paramref name="x"/> (returns x when the attribute is absent).</summary>
    private Value ApplyModifier(OnnxNode n, string attribute, Value x)
    {
        if (!n.Attributes.TryGetValue(attribute, out var graphAttribute) || graphAttribute.G is not { } g)
        {
            return x;
        }

        if (g.Inputs.Count != 1 || g.Outputs.Count != 1)
        {
            throw new InvalidDataException($"{n.OpType} {attribute} must have one input and one output");
        }

        _env[g.Inputs[0].Name] = TranslatedTensor.Dyn(x);
        RunSubgraph(g, $"{n.OpType} {attribute}");
        return AsValue(Get(g.Outputs[0].Name));
    }

    /// <summary>
    /// LinearAttention (opset 27): query/key/value [B, T, H·D], optional past_state [B, Hkv, Dk, Dv], decay and beta as the update rule
    /// requires (linear, gated, delta, gated_delta); outputs the attention output and present_state. The IR LinearAttention op (CPU).
    /// </summary>
    private void OnnxLinearAttention(OnnxNode n, TranslatedTensor?[] ins)
    {
        string rule = n.Str("update_rule", "gated_delta");
        bool gated = rule is "gated" or "gated_delta";
        bool delta = rule is "delta" or "gated_delta";
        if (rule is not ("linear" or "gated" or "delta" or "gated_delta"))
        {
            throw new InvalidDataException($"LinearAttention update_rule '{rule}'");
        }

        long qHeads = n.Int("q_num_heads", 0);
        long kvHeads = n.Int("kv_num_heads", 0);
        var key = X(ins, 1);
        long dk = key.Shape[2].Fixed / Math.Max(1, kvHeads);
        float scale = n.Float("scale", 0f) is var s && s != 0 ? s : 1f / MathF.Sqrt(dk);
        var operands = new List<Value> { X(ins, 0), key, X(ins, 2) };
        bool hasState = Opt(ins, 3) is not null;
        if (hasState)
        {
            operands.Add(X(ins, 3));
        }

        if (gated)
        {
            operands.Add(X(ins, 4));
        }

        if (delta)
        {
            operands.Add(X(ins, 5));
        }

        var attributes = Attributes.From(("q_heads", qHeads), ("kv_heads", kvHeads), ("scale", scale), ("state", hasState), ("gated", gated), ("delta", delta));
        var outputs = _b.Emit("LinearAttention", operands, attributes, Nm(n));
        Heavy();
        SetDyn(n, 0, outputs[0]);
        if (n.Outputs.Count > 1 && n.Outputs[1] != "")
        {
            SetDyn(n, 1, outputs[1]);
        }
    }

    /// <summary>[B, S, H·D] → [B, H, S, D].</summary>
    private Value SplitHeads(Value x, long heads)
    {
        long s = x.Shape[1].Fixed;
        long width = x.Shape[2].Fixed;
        if (width % heads != 0)
        {
            throw new InvalidDataException($"hidden size {width} is not a multiple of {heads} heads");
        }

        return Permute(Rs(x, 0, s, heads, width / heads), [0, 2, 1, 3]);
    }

    /// <summary>
    /// The additive score bias of an Attention node (null when nothing is masked) and, when some query row may see no key at all, the
    /// 0/1 mask of rows that see at least one (null when every row provably sees one). Static parts (causal, windows) are folded into one
    /// constant; the attention mask and nonpad_kv_seqlen add runtime parts.
    /// </summary>
    private (Value? Bias, Value? Visible) AttentionBias(OnnxNode n, TranslatedTensor?[] ins, long s, long t, long past)
    {
        bool causal = n.Int("is_causal", 0) != 0;
        long left = n.Int("left_window_size", -1);
        long right = n.Int("right_window_size", -1);
        bool windowed = left >= 0 || right >= 0;
        var nonpad = Opt(ins, 6);
        if (nonpad is not null && past > 0)
        {
            throw new InvalidDataException("Attention: nonpad_kv_seqlen cannot be combined with past key/value");
        }

        Value? additive = null;   // float attention mask
        Value? allowed = null;    // runtime 0/1 visibility
        if (Opt(ins, 3) is { } maskInput)
        {
            var m = AsValue(maskInput);
            long width = m.Shape[m.Rank - 1].Fixed;
            bool isBool = IsBoolTensor(n.Inputs[3], maskInput);
            if (width < t)
            {
                var pads = new long[2 * m.Rank];
                pads[2 * m.Rank - 1] = t - width;
                m = Emit("Pad", [m], Attributes.From(("pads", pads), ("mode", "constant"), ("value", isBool ? 0f : float.NegativeInfinity)));
            }
            else if (width > t)
            {
                throw new InvalidDataException($"Attention mask covers {width} keys, more than the {t} keys");
            }

            if (isBool)
            {
                allowed = m;
            }
            else
            {
                additive = m;
            }
        }

        // Static visibility [S, T] of causal and window masking with the internal (scalar) offset.
        float[]? fixedAllowed = null;
        if ((causal || windowed) && nonpad is null)
        {
            fixedAllowed = new float[s * t];
            for (long i = 0; i < s; i++)
            {
                for (long j = 0; j < t; j++)
                {
                    long diff = i + past - j;
                    bool sees = (!causal || diff >= 0) && (left < 0 || diff <= left) && (right < 0 || -diff <= right);
                    fixedAllowed[i * t + j] = sees ? 1f : 0f;
                }
            }
        }

        if (nonpad is not null)
        {
            // External cache: key j of batch row b is valid iff j < nonpad[b]; the causal / window frontier is offset by nonpad[b] - S.
            var valid = Rs(AsValue(nonpad), 0, 1, 1, 1);
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

            Value GridConstant(string suffix, float[] values) => _b.Constant(Nm(n) + suffix, DataType.F32, new Shape(s, t), ToBytes(values));
            var keys = _b.Constant(Nm(n) + ".keys", DataType.F32, new Shape(t), ToBytes([.. Enumerable.Range(0, (int)t).Select(j => (float)j)]));
            allowed = AndMask(allowed, _b.Binary("Less", keys, valid));
            if (causal)
            {
                // j <= i + nonpad - S  ⇔  j - i + S <= nonpad
                allowed = AndMask(allowed, _b.Binary("LessEqual", GridConstant(".causal", Grid((i, j) => j - i + s)), valid));
            }

            if (left >= 0)
            {
                // i + nonpad - S - j <= left  ⇔  nonpad <= left - i + j + S
                allowed = AndMask(allowed, _b.Binary("LessEqual", valid, GridConstant(".left", Grid((i, j) => left - i + j + s))));
            }

            if (right >= 0)
            {
                // j - (i + nonpad - S) <= right  ⇔  j - i + S - right <= nonpad
                allowed = AndMask(allowed, _b.Binary("LessEqual", GridConstant(".right", Grid((i, j) => j - i + s - right)), valid));
            }
        }

        if (allowed is null && additive is null)
        {
            if (fixedAllowed is null)
            {
                return (null, null);
            }

            // Fully static: an additive constant, and a constant row guard only if some row sees no key.
            var bias = _b.Constant(Nm(n) + ".bias", DataType.F32, new Shape(s, t), ToBytes([.. fixedAllowed.Select(a => a != 0 ? 0f : float.NegativeInfinity)]));
            var rows = Enumerable.Range(0, (int)s).Select(i => fixedAllowed.AsSpan(i * (int)t, (int)t).Contains(1f) ? 1f : 0f).ToArray();
            return (bias, rows.All(r => r != 0) ? null : _b.Constant(Nm(n) + ".rows", DataType.F32, new Shape(s, 1), ToBytes(rows)));
        }

        if (fixedAllowed is not null)
        {
            allowed = AndMask(allowed, _b.Constant(Nm(n) + ".visible", DataType.F32, new Shape(s, t), ToBytes(fixedAllowed)));
        }

        Value? total = additive;
        if (allowed is not null)
        {
            var visibility = Emit("Where", [allowed, ConstValue(OnnxConst.Scalar(0f), "attn.zero"), ConstValue(OnnxConst.Scalar(float.NegativeInfinity), "attn.neginf")]);
            total = total is null ? visibility : _b.Add(total, visibility);
        }

        // A row sees some key iff the maximum of its bias is above -inf.
        var rowMax = Emit("Reduce", [total!], Attributes.From(("mode", "max"), ("axes", L(-1)), ("keepdims", true)));
        var visible = _b.Binary("Greater", rowMax, ConstValue(OnnxConst.Scalar(float.NegativeInfinity), "attn.neginf"));
        return (total, visible);
    }

    private Value AndMask(Value? a, Value b) => a is null ? b : _b.Mul(a, b);

    /// <summary>
    /// Whether an ONNX value is boolean. Bool tensors travel as f32 0/1 like float tensors, so this consults the constant, the declared
    /// value types and finally the producing operator.
    /// </summary>
    private bool IsBoolTensor(string name, TranslatedTensor tensor)
    {
        if (tensor.C is { } c)
        {
            return c.ElemType == OnnxType.Bool;
        }

        var info = _graph.Inputs.Concat(_graph.ValueInfo).Concat(_graph.Outputs).FirstOrDefault(i => i.Name == name && i.ElemType != 0);
        if (info is not null)
        {
            return info.ElemType == OnnxType.Bool;
        }

        var producer = _graph.Nodes.FirstOrDefault(p => p.Outputs.Contains(name));
        return producer is not null
            && (s_boolProducers.Contains(producer.OpType) || (producer.OpType == "Cast" && producer.Int("to", 0) == OnnxType.Bool));
    }
}
