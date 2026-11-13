namespace Sharplight.Gguf;

/// <summary>llama.cpp <c>ernie4_5</c> (dense ERNIE 4.5): the llama graph with interleaved rotary pairs (stored unpermuted).</summary>
internal sealed class Ernie45GgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["ernie4_5"];

    /// <inheritdoc/>
    protected override string RopeStyle => "gptj";
}
