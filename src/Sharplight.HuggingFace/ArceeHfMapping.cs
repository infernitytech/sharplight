namespace Sharplight.HuggingFace;

/// <summary>Hugging Face <c>arcee</c> (AFM-4.5B): the Llama decoder with a plain squared-ReLU MLP (<c>up_proj</c>/<c>down_proj</c>); YaRN rope scaling.</summary>
internal sealed class ArceeHfMapping : HfDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["arcee"];

    /// <inheritdoc/>
    protected override bool Gated => false;
}
