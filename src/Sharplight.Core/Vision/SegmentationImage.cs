namespace Sharplight.Vision;

/// <summary>
/// An image encoded by an <see cref="IImageSegmenter"/>: the expensive image stage is done once, any number of prompts can then be segmented
/// from it quickly (<see cref="IImageSegmenter.SegmentAsync(SegmentationImage, SegmentationPrompt, SegmentationOptions?, CancellationToken)"/>).
/// </summary>
public sealed class SegmentationImage
{
    internal SegmentationImage(object owner, int width, int height, float[] embedding, float[] features0, float[] features1)
    {
        Owner = owner;
        Width = width;
        Height = height;
        Embedding = embedding;
        Features0 = features0;
        Features1 = features1;
    }

    /// <summary>Width of the encoded image in pixels.</summary>
    public int Width { get; }

    /// <summary>Height of the encoded image in pixels.</summary>
    public int Height { get; }

    internal object Owner { get; }

    internal float[] Embedding { get; }

    internal float[] Features0 { get; }

    internal float[] Features1 { get; }
}
