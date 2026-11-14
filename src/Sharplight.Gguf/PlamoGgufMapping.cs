namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>plamo</c> (PLaMo-13B): attention and SwiGLU MLP run in parallel from one RMS norm (x + attn(n(x)) + mlp(n(x))), neox rotary.
/// The converter regrouped the query heads so that llama.cpp's consecutive key/value head sharing applies, which is the canonical grouping too.
/// </summary>
internal sealed class PlamoGgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["plamo"];

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
        => md[GgufDecoderHyperParameters.Prefix + "residual.parallel"] = true;
}
