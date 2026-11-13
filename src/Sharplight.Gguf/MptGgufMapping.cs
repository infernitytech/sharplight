using Sharplight.Models;

namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>mpt</c> (MPT-7B/30B): ALiBi with the file's maximum bias (<c>attention.max_alibi_bias</c>; none stored means no bias and the learned
/// <c>position_embd</c> table), layer norm (without bias when the file has none), a fused q/k/v, optional clamping (<c>attention.clamp_kqv</c>) and a plain
/// GELU MLP. Files with q/k layer norms are rejected.
/// </summary>
internal sealed class MptGgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["mpt"];

    /// <inheritdoc/>
    protected override bool LayerNorm => true;

    /// <inheritdoc/>
    protected override bool Gated => false;

    /// <inheritdoc/>
    protected override string Activation => "gelu_tanh";

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        string prefix = GgufDecoderHyperParameters.Prefix;
        md[prefix + "rope.layers"] = new int[g.Int("block_count")];
        if (g.Float("attention.max_alibi_bias", 0f) is > 0f and var maxBias)
        {
            md[prefix + "attention.alibi_max_bias"] = maxBias;
        }

        if (g.Float("attention.clamp_kqv", 0f) is > 0f and var clamp)
        {
            md[prefix + "attention.clip_qkv"] = clamp;
        }
    }

    /// <inheritdoc/>
    protected override void RefineFamilyHyperParameters(Dictionary<string, object> md, GgufMetadataView g, IReadOnlyDictionary<string, WeightInfo> weights)
    {
        if (weights.Keys.Any(name => name.EndsWith(".attention.q_norm.weight", StringComparison.Ordinal)))
        {
            throw new NotSupportedException("mpt: q/k layer norms are not supported.");
        }
    }
}
