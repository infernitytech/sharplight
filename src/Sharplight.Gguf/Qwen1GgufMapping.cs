namespace Sharplight.Gguf;

/// <summary>llama.cpp <c>qwen</c> (Qwen v1): multi-head attention with a fused biased q/k/v tensor; the gate and up projections are half the stored feed-forward length.</summary>
internal sealed class Qwen1GgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["qwen"];

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        md[GgufDecoderHyperParameters.Prefix + "ffn_hidden_size"] = g.Int("feed_forward_length") / 2;
    }
}
