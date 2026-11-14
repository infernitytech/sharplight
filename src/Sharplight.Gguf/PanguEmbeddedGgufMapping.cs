namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>pangu-embedded</c> (openPangu-Embedded-1B/7B, chat models for edge devices): the Llama layer with neox rotary, optional q/k/v biases,
/// a biased output projection and an optional output bias.
/// </summary>
internal sealed class PanguEmbeddedGgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["pangu-embedded"];
}
