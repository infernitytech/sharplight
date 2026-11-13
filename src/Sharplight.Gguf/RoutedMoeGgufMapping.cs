namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp MoE decoders whose layers are the llama layer with q/k norms (when present) and the generic routed-expert block, all of
/// whose routing the file states (<c>expert_gating_func</c>, <c>expert_weights_norm</c>, <c>expert_weights_scale</c>,
/// <c>expert_group_count</c>, selection bias, shared experts): <c>bailingmoe2</c> (Ling / Ring 2.0), <c>dots1</c> (dots.llm1) and
/// <c>minimax-m2</c> (MiniMax-M2: full-width q/k norms, always renormalized) and <c>hy_v3</c> (Hunyuan 3: sigmoid unless the file says otherwise).
/// </summary>
internal sealed class RoutedMoeGgufMapping : GgufDecoderFamilyMapping
{
    // hy_v3 names its selection bias without the ".bias" suffix.
    private static readonly Dictionary<string, string> s_tensors = new(StringComparer.Ordinal) { ["exp_probs_b"] = "mlp.router_bias" };

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string>? LayerTensorOverrides => s_tensors;

    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["bailingmoe2", "dots1", "minimax-m2", "hy_v3"];

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        if (g.GgufArchitecture == "hy_v3" && g.Int("expert_gating_func", 0) == 0)
        {
            md[GgufDecoderHyperParameters.Prefix + "moe.router_activation"] = "sigmoid";   // llama.cpp's default for this architecture
        }

        if (g.GgufArchitecture == "minimax-m2")
        {
            md[GgufDecoderHyperParameters.Prefix + "moe.normalize_weights"] = true;   // llama.cpp hard-codes the renormalization
        }
    }
}
