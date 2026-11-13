namespace Sharplight.HuggingFace;

/// <summary>Hugging Face <c>ernie4_5</c> (dense ERNIE 4.5): the Llama decoder with interleaved rotary pairs.</summary>
internal sealed class Ernie45HfMapping : HfDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["ernie4_5"];

    /// <inheritdoc/>
    protected override string RopeStyle => "gptj";
}
