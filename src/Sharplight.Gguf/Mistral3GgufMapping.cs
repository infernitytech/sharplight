namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>mistral3</c> (Ministral 3, Mistral Small 3.x text): the llama layer (q/k rows stored permuted for llama.cpp's interleaved
/// rope) with YaRN rope scaling and, when the file has <c>attention.temperature_scale</c>, Llama-4-style query temperature on every layer
/// with the original YaRN context as its length and no position offset.
/// </summary>
internal sealed class Mistral3GgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["mistral3"];

    /// <inheritdoc/>
    protected override bool PermutesQk => true;

    /// <inheritdoc/>
    /// <exception cref="InvalidDataException">The temperature scale is set without an original context length.</exception>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        float scale = g.Float("attention.temperature_scale", 0f);
        if (scale == 0f)
        {
            return;
        }

        int length = g.Int("rope.scaling.original_context_length", 0);
        if (length <= 0)
        {
            throw new InvalidDataException("mistral3: attention.temperature_scale needs rope.scaling.original_context_length.");
        }

        string prefix = GgufDecoderHyperParameters.Prefix;
        md[prefix + "attention.temperature_scale"] = scale;
        md[prefix + "attention.temperature_length"] = (float)length;
    }
}
