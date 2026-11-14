namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>qwen2moe</c> (Qwen1.5-MoE, Qwen2-57B-A14B): the Qwen2 layer with softmax routing (selected weights not renormalized) and a
/// shared expert gated per token by sigmoid(h · <c>ffn_gate_inp_shexp</c>).
/// </summary>
internal sealed class Qwen2MoeGgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["qwen2moe"];

    /// <inheritdoc/>
    protected override bool NormalizesExpertWeights => false;
}
