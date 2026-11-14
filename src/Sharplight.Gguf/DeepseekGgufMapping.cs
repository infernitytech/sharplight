namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>deepseek</c> (DeepSeek-MoE 16B, DeepSeek V1): the Llama layer (q/k rows permuted for interleaved rope) with leading dense layers
/// (<c>leading_dense_block_count</c>), then softmax-routed SwiGLU experts whose top-k weights are not renormalized (scaled by
/// <c>expert_weights_scale</c>) plus an always-on shared SwiGLU expert. The weights tell the dense and MoE layers apart.
/// </summary>
internal sealed class DeepseekGgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["deepseek"];

    /// <inheritdoc/>
    protected override bool PermutesQk => true;

    /// <inheritdoc/>
    protected override bool NormalizesExpertWeights => false;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        // llama.cpp hard-wires the plain softmax top-k weights for this architecture, whatever the file says.
        md[GgufDecoderHyperParameters.Prefix + "moe.normalize_weights"] = false;
    }
}
