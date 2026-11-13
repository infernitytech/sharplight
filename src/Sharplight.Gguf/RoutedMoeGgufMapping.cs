namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp MoE decoders whose layers are the llama layer with q/k norms (when present) and the generic routed-expert block, all of
/// whose routing the file states (<c>expert_gating_func</c>, <c>expert_weights_norm</c>, <c>expert_weights_scale</c>,
/// <c>expert_group_count</c>, selection bias, shared experts): <c>bailingmoe2</c> (Ling / Ring 2.0), <c>dots1</c> (dots.llm1) and
/// <c>minimax-m2</c> (MiniMax-M2: full-width q/k norms, always renormalized).
/// </summary>
internal sealed class RoutedMoeGgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["bailingmoe2", "dots1", "minimax-m2"];

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        if (g.GgufArchitecture == "minimax-m2")
        {
            md[GgufDecoderHyperParameters.Prefix + "moe.normalize_weights"] = true;   // llama.cpp hard-codes the renormalization
        }
    }
}
