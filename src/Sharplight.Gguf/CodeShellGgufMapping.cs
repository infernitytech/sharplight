using Sharplight.Models;

namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>codeshell</c> (CodeShell-7B): layer norm with bias, grouped-query attention with a (possibly fused) biased q/k/v projection, neox rotary,
/// a plain biased GELU MLP. A file without <c>token_embd</c> embeds through the output projection.
/// </summary>
internal sealed class CodeShellGgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["codeshell"];

    /// <inheritdoc/>
    protected override bool LayerNorm => true;

    /// <inheritdoc/>
    protected override bool Gated => false;

    /// <inheritdoc/>
    protected override string Activation => "gelu_tanh";

    /// <inheritdoc/>
    protected override IEnumerable<WeightInfo> FamilySyntheticWeights(GgufMetadataView g, IReadOnlyDictionary<string, WeightInfo> weights)
    {
        if (!weights.ContainsKey("token_embedding.weight") && weights.TryGetValue("lm_head.weight", out var head))
        {
            yield return head.As("token_embedding.weight");
        }
    }
}
