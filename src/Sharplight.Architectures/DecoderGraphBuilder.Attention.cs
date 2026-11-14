using Sharplight.Graph;
using Sharplight.Models;

namespace Sharplight.Architectures;

internal sealed partial class DecoderGraphBuilder
{
    /// <summary>
    /// Emits the attention block of one layer. With <paramref name="rowSpans"/> the layer attends within packed sequences
    /// (keys and values stay in <paramref name="packedKv"/>); otherwise it reads and writes the KV cache.
    /// </summary>
    private Value Attention(GraphBuilder b, Value h, string p, int il, Value positions, Value? freqFactors, Value? rowSpans,
        Dictionary<int, (Value K, Value V)> packedKv)
    {
        int headDim = _hp.HeadDimOf(il);
        int heads = _hp.HeadsOf(il);
        var rotate = RotationOf(b, il, positions, freqFactors);

        var q = ClipQkv(b, b.Linear(h, p + "attention.q.weight", p + "attention.q.bias"));
        q = SplitHeads(b, q, b.TryWeight(p + "attention.q_norm.weight"), il, heads, headDim, rotate);
        if (_hp.TemperatureOn(il))
        {
            q = b.Mul(q, AttentionTemperature.Build(b, positions, _hp.TemperatureScale, _hp.TemperatureLength, _hp.TemperatureOffset), $"{p}q_tempered");
        }

        if (_hp.KvSourceOf(il) == il)
        {
            var (k, v) = ProjectKeyValue(b, h, p, il, rotate);
            if (rowSpans is not null)
            {
                packedKv[il] = (k, v);
            }
            else
            {
                b.CacheWrite(b.State(_kCache[il]), k);
                b.CacheWrite(b.State(_vCache[il]), v);
            }
        }

        if (rowSpans is not null)
        {
            // Packed sequences: causal within each sequence; sliding windows are not applied (inputs are short).
            var (packedK, packedV) = packedKv[_hp.KvSourceOf(il)];
            var packed = PackedSequences.Attention(b, q, packedK, packedV, rowSpans, _hp.AttnScale, causal: true, _hp.SoftCap, $"{p}attn", _hp.AlibiSlopes(heads));
            return AttentionOutput(b, packed, h, p, heads, _hp.ValueDimOf(il));
        }

        return AttentionOutput(b, CachedAttention(b, q, p, il), h, p, heads, _hp.ValueDimOf(il));
    }

    /// <summary>Clamps a q/k/v projection to ±<c>attention.clip_qkv</c> (OLMo 1) when the model asks for it.</summary>
    private Value ClipQkv(GraphBuilder b, Value projection)
        => _hp.ClipQkv > 0 ? b.Emit1("Clip", [projection], Attributes.From(("min", -_hp.ClipQkv), ("max", _hp.ClipQkv))) : projection;

    /// <summary>Returns the rotary embedding to apply to q and k of a layer (identity for NoPE layers).</summary>
    private Func<Value, Value> RotationOf(GraphBuilder b, int il, Value positions, Value? freqFactors)
    {
        bool sliding = _hp.Sliding(il);
        if (!_hp.RopeOn(il))
        {
            return t => t;
        }

        int ropeDims = sliding ? _hp.RopeLocalDims : _hp.RopeDims;
        var rope = Attributes.From(("dims", ropeDims), ("base", sliding ? _hp.RopeLocalBase : _hp.RopeBase), ("style", _hp.RopeStyle));
        bool scaled = !sliding || _hp.RopeLocalFreqFactors;
        if (scaled && _hp.RopeAttentionScale != 1f)
        {
            rope = rope.With("attn_factor", _hp.RopeAttentionScale);
        }

        return t => b.Emit1("Rope", freqFactors is { } ff && scaled ? [t, positions, ff] : [t, positions], rope);
    }

    /// <summary>
    /// Splits a projection into heads, applies the q/k norm and the rotary embedding (norm first, or rotation first when the model
    /// norms after rope). The norm weight covers either the whole projection (heads · head dimension entries, applied before the split)
    /// or one head (head dimension entries, applied after it); without a weight an unweighted norm is applied when the model asks for one.
    /// </summary>
    private Value SplitHeads(GraphBuilder b, Value projection, Value? normWeight, int il, int heads, int headDim, Func<Value, Value> rotate)
    {
        var x = projection;
        if (normWeight is { } wholeNorm && wholeNorm.Shape[0].Fixed != headDim)
        {
            x = b.RmsNorm(x, wholeNorm, _hp.Eps);
        }

        x = b.Reshape(x, 0, heads, headDim);
        if (_hp.QkNormAfterRope)
        {
            return NormHead(b, rotate(x), normWeight, il, headDim);
        }

        return rotate(NormHead(b, x, normWeight, il, headDim));
    }

    // The per-head part of the q/k norm: a head-sized weight, or an unweighted norm when the model asks for one and has no weight.
    private Value NormHead(GraphBuilder b, Value x, Value? normWeight, int il, int headDim)
    {
        if (normWeight is { } headNorm && headNorm.Shape[0].Fixed == headDim)
        {
            return b.RmsNorm(x, headNorm, _hp.Eps);
        }

        return normWeight is null && _hp.QkNormUnweightedOn(il) ? b.RmsNorm(x, null, _hp.Eps) : x;
    }

    /// <summary>Projects the layer input to rotated keys and values (k = v layers reuse the key projection for values).</summary>
    private (Value K, Value V) ProjectKeyValue(GraphBuilder b, Value h, string p, int il, Func<Value, Value> rotate)
    {
        int headDim = _hp.HeadDimOf(il);
        int kvHeads = _hp.KvHeadsOf(il);
        var keyProjection = ClipQkv(b, b.Linear(h, p + "attention.k.weight", p + "attention.k.bias"));
        var valueProjection = b.HasWeight(p + "attention.v.weight") ? ClipQkv(b, b.Linear(h, p + "attention.v.weight", p + "attention.v.bias")) : keyProjection;
        var v = b.Reshape(valueProjection, 0, kvHeads, _hp.ValueDimOf(il));
        if (_hp.VNorm)
        {
            v = b.RmsNorm(v, null, _hp.Eps);
        }

        var k = SplitHeads(b, keyProjection, b.TryWeight(p + "attention.k_norm.weight"), il, kvHeads, headDim, rotate);
        return (k, v);
    }

    /// <summary>Attention of the queries against the layer's KV cache, honouring window, soft cap and sinks.</summary>
    private Value CachedAttention(GraphBuilder b, Value q, string p, int il)
    {
        var attrs = new List<(string, object)> { ("scale", _hp.AttnScale), ("offset", WellKnown.PastScalar) };
        if (_hp.Windowed(il))
        {
            attrs.Add(("window", _hp.Window));
        }

        if (_hp.Chunked(il))
        {
            attrs.Add(("chunk", _hp.ChunkSize));
        }

        if (_hp.SoftCap > 0)
        {
            attrs.Add(("softcap", _hp.SoftCap));
        }

        if (_hp.AlibiSlopes(_hp.HeadsOf(il)) is { } slopes)
        {
            attrs.Add(("alibi_slopes", slopes));
        }

        b.RuntimeScalar(WellKnown.PastScalar);
        Value[] attnIn = b.TryWeight(p + "attention.sinks") is { } sinks
            ? [q, b.State(_kCache[il]), b.State(_vCache[il]), sinks]
            : [q, b.State(_kCache[il]), b.State(_vCache[il])];
        if (attnIn.Length > 3)
        {
            attrs.Add(("sinks", true));
        }

        return b.Emit1("CachedAttention", attnIn, Attributes.From([.. attrs]), $"{p}attn");
    }

    /// <summary>
    /// Applies the optional value scale and output gate to the attention output and projects it. The gate is a projection of the layer input
    /// through the sigmoid (default) or softplus; a gate as wide as the head count has one scalar per head, a wider one a scalar per channel.
    /// </summary>
    private Value AttentionOutput(GraphBuilder b, Value attention, Value layerInput, string p, int heads, int valueDim)
    {
        var scaled = _hp.ValueScale != 1f ? b.Scale(attention, _hp.ValueScale, 0f) : attention;
        if (b.HasWeight(p + "attention.gate.weight"))
        {
            var projection = b.Linear(layerInput, p + "attention.gate.weight");
            var gate = _hp.GateActivation == "softplus" ? b.Softplus(projection) : b.Sigmoid(projection);
            bool perHead = b.Weight(p + "attention.gate.weight").Shape[0].Fixed == heads;
            scaled = b.Mul(scaled, perHead ? b.Reshape(gate, 0, heads, 1) : b.Reshape(gate, 0, heads, valueDim));
        }

        var flat = b.Reshape(scaled, 0, heads * valueDim);
        return b.Linear(SubNorm(b, flat, p + "attention.sub_norm.weight"), p + "attention.o.weight", p + "attention.o.bias", $"{p}attn_out");
    }
}
