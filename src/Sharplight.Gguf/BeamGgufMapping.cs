namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>beam</c>: sliding-window layers from the per-layer pattern with rotary embedding on their own base (the full-attention layers
/// are NoPE and scale their queries by the power-form temperature (1 + (pos + 1) / L)^β instead), per-head q/k norms, a per-channel sigmoid
/// output gate with one bias per head, sandwich norms, norm-preserving residual sums, leading dense layers followed by routed SwiGLU experts
/// with a selection bias (sigmoid unless the file says otherwise), and an optional final logit soft cap.
/// </summary>
internal sealed class BeamGgufMapping : GgufDecoderFamilyMapping
{
    private static readonly Dictionary<string, string> s_tensors = new(StringComparer.Ordinal)
    {
        ["attn_gate.weight"] = "attention.gate.weight",
        ["attn_gate.bias"] = "attention.gate.bias",
    };

    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["beam"];

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string>? LayerTensorOverrides => s_tensors;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        string prefix = GgufDecoderHyperParameters.Prefix;
        int layerCount = g.Int("block_count");
        int[] sliding = GgufMappingHelpers.SlidingLayers(g, layerCount, 0, denseFirst: false);
        md[prefix + "attention.sliding_window"] = g.Int("attention.sliding_window");
        md[prefix + "attention.sliding_layers"] = sliding;
        md[prefix + "rope.layers"] = sliding;
        md[prefix + "rope.local_base"] = g.Float("rope.freq_base_swa", g.Float("rope.freq_base", GgufMappingHelpers.DefaultRopeBase));
        md[prefix + "attention.temperature_scale"] = g.Float("attention.temperature_scale", 0.1f);
        md[prefix + "attention.temperature_length"] = (float)g.Int("attention.temperature_length", 2048);
        md[prefix + "attention.temperature_offset"] = 1f;
        md[prefix + "attention.temperature_power"] = true;
        md[prefix + "attention.temperature_layers"] = sliding.Select(s => 1 - s).ToArray();
        md[prefix + "residual.norm_preserving"] = true;
        if (g.Int("expert_gating_func", 0) == 0)
        {
            md[prefix + "moe.router_activation"] = "sigmoid";   // llama.cpp's default for this architecture
        }

        if (g.Float("final_logit_softcapping", 0f) is > 0f and var cap)
        {
            md[prefix + "final_logit_softcap"] = cap;
        }
    }
}
