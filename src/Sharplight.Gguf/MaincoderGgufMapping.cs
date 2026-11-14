namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>maincoder</c>: a Llama layer whose per-head q/k RMS norms follow the rotary embedding, which rotates interleaved pairs (llama.cpp's
/// "normal" rope on rows the converter did not permute).
/// </summary>
internal sealed class MaincoderGgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["maincoder"];

    /// <inheritdoc/>
    protected override string RopeStyle => "gptj";

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
        => md[GgufDecoderHyperParameters.Prefix + "attention.qk_norm_after_rope"] = true;
}
