namespace Sharplight.Vision;

/// <summary>Per-request settings of <see cref="IImageEmbedder"/>.</summary>
public sealed class ImageEmbeddingOptions
{
    /// <summary>Whether the embedding is scaled to unit length (the form similarity search and zero-shot scoring use).</summary>
    public bool Normalize { get; set; } = true;
}
