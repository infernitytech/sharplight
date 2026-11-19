using Sharplight.Vision;

namespace Sharplight.Llm.Serving;

/// <summary>An image of an <see cref="ImagePrompts"/> prompt: the preprocessed pixels its vision encoder turns into rows.</summary>
internal sealed class ImageMedia(ImagePrompts owner, PreprocessedImage image, UInt128 hash) : MediaContent(hash, image.TokenCount)
{
    /// <summary>The prompt builder (and encoder) the image belongs to.</summary>
    public ImagePrompts Owner { get; } = owner;

    /// <summary>The preprocessed image.</summary>
    public PreprocessedImage Image { get; } = image;

    /// <inheritdoc/>
    public override IMediaEncoder Encoder => Owner;
}
