namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>bailingmoe</c> (Ling-lite / Ling-plus, Bailing MoE V1): the Llama layer (q/k rows permuted for interleaved rope) with a head size
/// of <c>rope.dimension_count</c>, softmax-routed SwiGLU experts (renormalized only when <c>expert_weights_norm</c> says so, scaled by
/// <c>expert_weights_scale</c>) plus an always-on shared SwiGLU expert.
/// </summary>
internal sealed class BailingMoeGgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["bailingmoe"];

    /// <inheritdoc/>
    protected override bool PermutesQk => true;

    /// <inheritdoc/>
    protected override bool NormalizesExpertWeights => false;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        if (g.Has("rope.dimension_count"))
        {
            md[GgufDecoderHyperParameters.Prefix + "attention.head_dim"] = g.Int("rope.dimension_count");
        }
    }
}
