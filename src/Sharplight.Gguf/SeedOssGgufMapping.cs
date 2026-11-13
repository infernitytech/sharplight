namespace Sharplight.Gguf;

/// <summary>llama.cpp <c>seed_oss</c> (Seed-OSS): the qwen2-like graph with q/k/v biases; its <c>post_attention_norm</c> tensor is the norm in front of the MLP.</summary>
internal sealed class SeedOssGgufMapping : GgufDecoderFamilyMapping
{
    private static readonly Dictionary<string, string> s_tensors = new(StringComparer.Ordinal)
    {
        ["post_attention_norm.weight"] = "post_attention_norm.weight",
    };

    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["seed_oss"];

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string>? LayerTensorOverrides => s_tensors;
}
