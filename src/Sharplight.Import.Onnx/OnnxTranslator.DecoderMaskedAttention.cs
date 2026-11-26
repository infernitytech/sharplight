using Sharplight.Graph;

namespace Sharplight.Import.Onnx;

// com.microsoft DecoderMaskedMultiHeadAttention (onnxruntime's one-token decoder attention of the beam-search / Whisper / T5 exports).
internal sealed partial class OnnxTranslator
{
    /// <summary>
    /// DecoderMaskedMultiHeadAttention: one query token [B, 1, H·D]. Self attention writes the new key/value [B, 1, H·D] into row p =
    /// past_sequence_length of the shared past buffers [B, H, M, D] (present = past with that row replaced, the other rows kept) and
    /// attends to rows 0..p; cross attention reads key/value already in heads [B, H, L, D] without a past. A raw key mask [B, p + 1] and an
    /// attention_bias [B or 1, H, 1, p + 1] (both padded with hidden keys to the buffer length) and a packed q/k/v bias are supported; do_rotary,
    /// beam search (cache_indirection) and the qk output are not.
    /// </summary>
    private void DecoderMaskedMultiHeadAttention(OnnxNode n, TranslatedTensor?[] ins)
    {
        if (n.Int("do_rotary", 0) != 0 || Opt(ins, 9) is not null)
        {
            throw new NotSupportedException("DecoderMaskedMultiHeadAttention with do_rotary or cache_indirection");
        }

        CheckUnusedOutputs(n, 3);
        string name = Nm(n);
        int heads = (int)n.Int("num_heads", 1);
        var q = X(ins, 0);
        if (q.Shape[1].Fixed != 1)
        {
            throw new NotSupportedException($"DecoderMaskedMultiHeadAttention with {q.Shape[1].Fixed} query tokens (one is decoded at a time)");
        }

        long hidden = q.Shape[2].Fixed;
        var key = X(ins, 1);
        bool cross = key.Rank == 4;
        Value k = key;
        Value v = X(ins, 2);
        if (Opt(ins, 10) is not null)
        {
            var bias = X(ins, 10);
            q = _b.Add(q, _b.Slice(bias, 0, 0, hidden));
            if (!cross)
            {
                k = _b.Add(k, _b.Slice(bias, 0, hidden, hidden));
                v = _b.Add(v, _b.Slice(bias, 0, 2 * hidden, v.Shape[2].Fixed));
            }
        }

        var qh = SplitHeads(q, heads);
        Value kh, vh;
        Value? visible = null;
        if (cross)
        {
            (kh, vh) = (k, v);
        }
        else
        {
            if (Opt(ins, 5) is null || Opt(ins, 6) is null || Opt(ins, 7) is null)
            {
                throw new NotSupportedException("DecoderMaskedMultiHeadAttention self attention needs past_key, past_value and past_sequence_length");
            }

            var pastKey = X(ins, 5);
            long t = pastKey.Shape[2].Fixed;
            var rows = F32Const([.. Enumerable.Range(0, (int)t).Select(j => (float)j)], name + ".rows");
            var p = Rs(AsValue(ins[7]!), 1, 1, 1, 1);
            var at = Rs(_b.Binary("Equal", rows, p), 1, 1, t, 1);   // [1, 1, T, 1]: the row the new token goes to
            Value Write(Value past, Value fresh)
                => _b.Add(_b.Mul(past, _b.Scale(at, -1f, 1f)), _b.Mul(at, SplitHeads(fresh, heads)));
            kh = Write(pastKey, k);
            vh = Write(X(ins, 6), v);
            visible = Rs(_b.Binary("LessEqual", rows, p), 1, 1, 1, t);   // keys 0..p
        }

        if (n.Outputs.Count > 1 && n.Outputs[1] != "")
        {
            SetDyn(n, 1, kh);
        }

        if (n.Outputs.Count > 2 && n.Outputs[2] != "")
        {
            SetDyn(n, 2, vh);
        }

        long keys = kh.Shape[2].Fixed;
        Value? additive = null;
        if (visible is not null)
        {
            // (1 - visible) · MinValue
            additive = _b.Scale(visible, -float.MinValue, float.MinValue);
        }

        if (Opt(ins, 3) is not null)
        {
            // A raw 0/1 key mask [B, p + 1]; the keys behind it (not written yet) are hidden.
            var mask = X(ins, 3);
            long width = mask.Shape[1].Fixed;
            if (mask.Rank != 2 || width > keys)
            {
                throw new NotSupportedException($"DecoderMaskedMultiHeadAttention mask_index of shape {mask.Shape} over {keys} keys");
            }

            if (width < keys)
            {
                mask = Emit("Pad", [mask], Attributes.From(("pads", L(0, 0, 0, keys - width)), ("mode", "constant"), ("value", 0f)));
            }

            float filter = n.Float("mask_filter_value", -10000f);
            var maskBias = _b.Scale(Rs(mask, 0, 1, 1, keys), -filter, filter);
            additive = additive is null ? maskBias : _b.Add(additive, maskBias);
        }

        if (Opt(ins, 4) is not null)
        {
            // [B or 1, H, 1, p + 1]: keys past the bias width are hidden anyway.
            var attentionBias = X(ins, 4);
            long width = attentionBias.Shape[3].Fixed;
            if (width < keys)
            {
                attentionBias = Emit("Pad", [attentionBias], Attributes.From(("pads", L(0, 0, 0, 0, 0, 0, 0, keys - width)), ("mode", "constant"), ("value", 0f)));
            }

            additive = additive is null ? attentionBias : _b.Add(additive, attentionBias);
        }

        float scale = n.Float("scale", 0f) is var sc && sc != 0 ? sc : 1f / MathF.Sqrt(qh.Shape[3].Fixed);
        var (y, _, _) = MultiHeadCore(qh, kh, vh, scale, additive, 0f, null, name, needScores: false);
        SetDyn(n, 0, MergeHeads(y, name));
    }
}
