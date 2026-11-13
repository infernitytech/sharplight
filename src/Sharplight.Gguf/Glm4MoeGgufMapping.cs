namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>glm4moe</c> (GLM-4.5 / 4.6 / 4.7, GLM-4.5-Air): biased q/k/v, optional per-head q/k norms, partial rotary embedding on
/// rotate-half pairs, leading dense layers, then sigmoid-scored experts with a selection bias, renormalization, a routed scaling factor
/// and a shared expert. llama.cpp's <c>post_attention_norm</c> is the norm in front of the MLP; the NextN (MTP) block is not used.
/// </summary>
internal sealed class Glm4MoeGgufMapping : GgufDecoderFamilyMapping
{
    private static readonly Dictionary<string, string> s_overrides = new(StringComparer.Ordinal)
    {
        ["post_attention_norm.weight"] = "post_attention_norm.weight",
    };

    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["glm4moe"];

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string>? LayerTensorOverrides => s_overrides;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        // llama.cpp's default when the file does not state a gating function.
        md.TryAdd(GgufDecoderHyperParameters.Prefix + "moe.router_activation", "sigmoid");
    }
}
