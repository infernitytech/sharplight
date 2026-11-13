namespace Sharplight.HuggingFace;

/// <summary>Hugging Face <c>seed_oss</c> (Seed-OSS): the Llama decoder with biased q/k/v projections.</summary>
internal sealed class SeedOssHfMapping : HfDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["seed_oss"];
}
