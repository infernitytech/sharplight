using Sharplight.Graph;
using Sharplight.Models;

namespace Sharplight.Architectures;

internal sealed partial class DecoderGraphBuilder
{
    /// <summary>
    /// Emits the attention block of one layer pass. With <paramref name="rowSpans"/> the layer attends within packed sequences
    /// (keys and values stay in <paramref name="packedKv"/>, keyed by cache slot); otherwise it reads and writes the KV cache of its slot.
    /// <paramref name="embeddings"/> are the model's input embeddings, which the value residual (<c>attention.value_residual.weight</c>) mixes in.
    /// </summary>
    private Value Attention(GraphBuilder b, Value h, string p, LayerSlot slot, Value positions, Value? freqFactors, Value? rowSpans,
        Dictionary<int, (Value K, Value V)> packedKv, Value? embeddings)
    {
        int il = slot.Layer;
        int headDim = _hp.HeadDimOf(il);
        int heads = _hp.HeadsOf(il);
        var rotate = RotationOf(b, il, positions, freqFactors);

        var q = ClipQkv(b, b.Linear(h, p + "attention.q.weight", p + "attention.q.bias"));
        q = SplitHeads(b, q, b.TryWeight(p + "attention.q_norm.weight"), b.TryWeight(p + "attention.q_norm.bias"), il, heads, headDim, rotate);
        if (_hp.TemperatureOn(il))
        {
            q = b.Mul(q, AttentionTemperature.Build(b, positions, _hp.TemperatureScale, _hp.TemperatureLength, _hp.TemperatureOffset, _hp.TemperaturePower), $"{p}q_tempered");
        }

        int sourceSlot = slot.Slot - il + _hp.KvSourceOf(il);
        if (sourceSlot == slot.Slot)
        {
            var (k, v) = ProjectKeyValue(b, h, ValueInput(b, h, p, embeddings), p, il, rotate);
            if (rowSpans is not null)
            {
                packedKv[slot.Slot] = (k, v);
            }
            else
            {
                b.CacheWrite(b.State(_kCache[slot.Slot]!), k);
                b.CacheWrite(b.State(_vCache[slot.Slot]!), v);
            }
        }

        if (rowSpans is not null)
        {
            // Packed sequences: causal within each sequence unless the model is bidirectional (then a sliding window is symmetric).
            var (packedK, packedV) = packedKv[sourceSlot];
            var packed = PackedSequences.Attention(b, q, packedK, packedV, rowSpans, _hp.AttnScale, _hp.Causal, _hp.SoftCap, $"{p}attn", _hp.AlibiSlopes(heads),
                _hp.Windowed(il) ? _hp.Window : 0);
            return AttentionOutput(b, packed, h, p, heads, _hp.ValueDimOf(il));
        }

        var selection = _blockIndexerCache[slot.Slot] is { } indexerKeys ? _blockIndexer!.Select(b, h, p, indexerKeys, rotate) : null;
        return AttentionOutput(b, CachedAttention(b, q, p, il, sourceSlot, selection), h, p, heads, _hp.ValueDimOf(il));
    }

    /// <summary>
    /// The input of the value projection: the normed layer input, or with a value residual weight λ (Ricursive) its interpolation
    /// x0 + (h − x0)·sigmoid(λ) towards the model's input embeddings x0.
    /// </summary>
    private static Value ValueInput(GraphBuilder b, Value h, string p, Value? embeddings)
    {
        if (b.TryWeight(p + "attention.value_residual.weight") is not { } lambda)
        {
            return h;
        }

        var x0 = embeddings ?? throw new InvalidOperationException($"{p}attention.value_residual needs the input embeddings.");
        return b.Add(x0, b.Mul(b.Sub(h, x0), b.Sigmoid(lambda)), $"{p}value_input");
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
        if (_hp.RopeSections.Length > 0 && positions.Rank == 2 && positions.Shape[-1].Fixed == DecoderHyperParameters.MRopeAxes)
        {
            // M-RoPE over (time, row, column, extra); the packed-sequence graphs carry one position axis (text: every axis equal, plain RoPE).
            rope = rope.With("sections", _hp.RopeSections).With("interleaved", _hp.RopeSectionsInterleaved);
        }

        bool scaled = !sliding || _hp.RopeLocalFreqFactors;
        if (scaled && _hp.RopeAttentionScale != 1f)
        {
            rope = rope.With("attn_factor", _hp.RopeAttentionScale);
        }

        return t => b.Emit1("Rope", freqFactors is { } ff && scaled ? [t, positions, ff] : [t, positions], rope);
    }

    /// <summary>
    /// Splits a projection into heads, applies the q/k norm and the rotary embedding (norm first, or rotation first when the model
    /// norms after rope). The norm weight covers either the whole projection (heads · head dimension entries in one axis, applied before the
    /// split), one head (head dimension entries, applied after it) or every head separately ([heads, head dimension], Chameleon); without a
    /// weight an unweighted norm is applied when the model asks for one. <paramref name="normBias"/> (layer norm only) has the weight's shape.
    /// </summary>
    private Value SplitHeads(GraphBuilder b, Value projection, Value? normWeight, Value? normBias, int il, int heads, int headDim, Func<Value, Value> rotate)
    {
        var x = projection;
        if (normWeight is { } wholeNorm && wholeNorm.Shape.Rank == 1 && wholeNorm.Shape.Elements() == (long)heads * headDim && wholeNorm.Shape[0].Fixed != headDim)
        {
            x = b.RmsNorm(x, wholeNorm, _hp.QkNormEps);
        }

        x = b.Reshape(x, 0, heads, headDim);
        if (_hp.QkNormAfterRope)
        {
            return NormHead(b, rotate(x), normWeight, normBias, il, headDim);
        }

        return rotate(NormHead(b, x, normWeight, normBias, il, headDim));
    }

    // The per-head part of the q/k norm: a head-sized weight, a [heads, head dimension] weight with one gain (and bias) per head, a [heads, 1]
    // gain per head after an unweighted norm (Talkie), or an unweighted norm when the model asks for one and has no weight.
    private Value NormHead(GraphBuilder b, Value x, Value? normWeight, Value? normBias, int il, int headDim)
    {
        if (normWeight is { } headNorm && headNorm.Shape.Rank == 1 && headNorm.Shape[0].Fixed == headDim)
        {
            return QkNorm(b, x, headNorm, normBias);
        }

        if (normWeight is { } perHead && perHead.Shape.Rank == 2 && perHead.Shape[1].Fixed == headDim)
        {
            var scaled = b.Mul(QkNorm(b, x, null, null), perHead);
            return normBias is null ? scaled : b.Add(scaled, normBias);
        }

        if (normWeight is { } headGain && headGain.Shape.Rank == 2 && headGain.Shape[1].Fixed == 1)
        {
            return b.Mul(QkNorm(b, x, null, null), headGain);
        }

        return normWeight is null && _hp.QkNormUnweightedOn(il) ? QkNorm(b, x, null, null) : x;
    }

    // One q/k norm over the head dimension: RMS norm, or a mean-centred layer norm when attention.qk_norm_type is "layer".
    private Value QkNorm(GraphBuilder b, Value x, Value? weight, Value? bias)
    {
        if (_hp.QkNormType != "layer")
        {
            return b.RmsNorm(x, weight, _hp.QkNormEps);
        }

        List<Value> operands = weight is null ? [x] : bias is null ? [x, weight] : [x, weight, bias];
        return b.Emit1("LayerNorm", operands, Attributes.From(("eps", _hp.QkNormEps)));
    }

    /// <summary>Projects the layer input to rotated keys and values (k = v layers reuse the key projection for values).</summary>
    private (Value K, Value V) ProjectKeyValue(GraphBuilder b, Value h, Value valueInput, string p, int il, Func<Value, Value> rotate)
    {
        int headDim = _hp.HeadDimOf(il);
        int kvHeads = _hp.KvHeadsOf(il);
        var keyProjection = ClipQkv(b, b.Linear(h, p + "attention.k.weight", p + "attention.k.bias"));
        var valueProjection = b.HasWeight(p + "attention.v.weight") ? ClipQkv(b, b.Linear(valueInput, p + "attention.v.weight", p + "attention.v.bias")) : keyProjection;
        var v = b.Reshape(valueProjection, 0, kvHeads, _hp.ValueDimOf(il));
        if (_hp.VNorm)
        {
            v = b.RmsNorm(v, null, _hp.Eps);
        }

        var k = SplitHeads(b, keyProjection, b.TryWeight(p + "attention.k_norm.weight"), b.TryWeight(p + "attention.k_norm.bias"), il, kvHeads, headDim, rotate);
        return (k, v);
    }

    /// <summary>
    /// Attention of the queries against the KV cache of <paramref name="cacheSlot"/>, honouring window, soft cap and sinks; with
    /// <paramref name="selection"/> (i32 [T, Hs, K]) only the selected key positions are attended.
    /// </summary>
    private Value CachedAttention(GraphBuilder b, Value q, string p, int il, int cacheSlot, Value? selection)
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
            ? [q, b.State(_kCache[cacheSlot]!), b.State(_vCache[cacheSlot]!), sinks]
            : [q, b.State(_kCache[cacheSlot]!), b.State(_vCache[cacheSlot]!)];
        if (attnIn.Length > 3)
        {
            attrs.Add(("sinks", true));
        }

        if (selection is not null)
        {
            attnIn = [.. attnIn, selection];
            attrs.Add(("selection", true));
        }

        return b.Emit1("CachedAttention", attnIn, Attributes.From([.. attrs]), $"{p}attn");
    }

    /// <summary>
    /// Applies the optional value scale and output gate to the attention output and projects it. The gate is a projection of the layer input
    /// through the sigmoid (default) or softplus; a gate as wide as the head count has one scalar per head, a wider one a scalar per channel;
    /// an optional <c>attention.gate.bias</c> is added before the activation (one entry per head is shared by the head's channels).
    /// </summary>
    private Value AttentionOutput(GraphBuilder b, Value attention, Value layerInput, string p, int heads, int valueDim)
    {
        var scaled = _hp.ValueScale != 1f ? b.Scale(attention, _hp.ValueScale, 0f) : attention;
        if (_hp.AttentionHeadNorm)
        {
            scaled = b.RmsNorm(b.Reshape(scaled, 0, heads, valueDim), null, _hp.Eps);   // weight-less norm of every head's output (Muse Spark)
        }

        if (b.HasWeight(p + "attention.gate.weight"))
        {
            bool perHead = b.Weight(p + "attention.gate.weight").Shape[0].Fixed == heads;
            var projection = perHead ? b.Reshape(b.Linear(layerInput, p + "attention.gate.weight"), 0, heads, 1)
                : b.Reshape(b.Linear(layerInput, p + "attention.gate.weight"), 0, heads, valueDim);
            if (b.TryWeight(p + "attention.gate.bias") is { } gateBias)
            {
                // A bias with one entry per head (Beam) is shared by the head's channels.
                projection = b.Add(projection, gateBias.Shape[0].Fixed == heads ? b.Reshape(gateBias, heads, 1) : b.Reshape(gateBias, heads, perHead ? 1 : valueDim));
            }

            scaled = b.Mul(scaled, _hp.GateActivation == "softplus" ? b.Softplus(projection) : b.Sigmoid(projection));
        }

        var flat = b.Reshape(scaled, 0, heads * valueDim);
        return b.Linear(SubNorm(b, flat, p + "attention.sub_norm.weight"), p + "attention.o.weight", p + "attention.o.bias", $"{p}attn_out");
    }
}
