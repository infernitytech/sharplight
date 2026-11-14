namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>limite</c> (Limite): weight-less RMS norms everywhere (the embeddings, q/k, the layer inputs, the output), sliding-window layers
/// with a partial rope in llama.cpp's normal (interleaved) pairing and global NoPE layers (default pattern 4), value embeddings
/// (<c>value_embd</c>) gated into the values per key-value head (<c>attn_ve_gate</c>), exclusive self attention (<c>attn_xsa_alpha</c>,
/// <c>attention.xsa_epsilon</c>), per-head sigmoid output gates (<c>attn_gate</c>), learned residual scales (<c>attn_resid_scale</c>,
/// <c>ffn_resid_scale</c>), dense cross-layer connections (<c>mudd_down</c> / <c>mudd_up</c>, <c>mudd.tap_count</c> / <c>mudd.tap_indices</c>) and a
/// sigmoid logit cap (<c>final_logit_sigmoid_capping</c>: scale, shift, temperature).
/// </summary>
internal sealed class LimiteGgufMapping : GgufDecoderFamilyMapping
{
    private static readonly Dictionary<string, string> s_layerTensors = new(StringComparer.Ordinal)
    {
        ["attn_gate.weight"] = "attention.gate.weight",
        ["attn_ve_gate.weight"] = "attention.value_embedding_gate.weight",
        ["attn_xsa_alpha.weight"] = "attention.exclusive.weight",
        ["attn_resid_scale.weight"] = "attention.residual_alpha",
        ["ffn_resid_scale.weight"] = "mlp.residual_alpha",
        ["mudd_down.weight"] = "dense.down.weight",
        ["mudd_up.weight"] = "dense.up.weight",
        ["mudd_up.bias"] = "dense.up.bias",
    };

    private static readonly Dictionary<string, string> s_globalTensors = new(StringComparer.Ordinal)
    {
        ["value_embd.weight"] = "value_embedding.weight",
    };

    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["limite"];

    /// <inheritdoc/>
    protected override string RopeStyle => "gptj";

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string>? LayerTensorOverrides => s_layerTensors;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string>? GlobalTensorOverrides => s_globalTensors;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        string prefix = GgufDecoderHyperParameters.Prefix;
        int layerCount = g.Int("block_count");
        int[] sliding = GgufMappingHelpers.SlidingLayers(g, layerCount, 4, denseFirst: false);
        md[prefix + "attention.sliding_window"] = g.Int("attention.sliding_window");
        md[prefix + "attention.sliding_layers"] = sliding;
        md[prefix + "rope.layers"] = sliding;
        md[prefix + "attention.scale"] = g.Float("attention.scale");
        md[prefix + "attention.qk_norm_unweighted"] = true;
        md[prefix + "attention.exclusive_eps"] = g.Float("attention.xsa_epsilon", 0f);
        md[prefix + "embedding_norm"] = true;
        if (g.Raw("final_logit_sigmoid_capping") is Array cap && cap.Length == 3 && Convert.ToSingle(cap.GetValue(0)) != 0f
            && Convert.ToSingle(cap.GetValue(2)) != 0f)
        {
            md[prefix + "logits.sigmoid_cap"] = cap.Cast<object>().Select(Convert.ToSingle).ToArray();
        }

        if (g.Int("mudd.tap_count", 0) is var taps and > 0 && g.Raw("mudd.tap_indices") is Array indices)
        {
            md[prefix + "dense_connections.tap_count"] = taps;
            md[prefix + "dense_connections.taps"] = indices.Cast<object>().Select(Convert.ToInt32).ToArray();
        }
    }
}
