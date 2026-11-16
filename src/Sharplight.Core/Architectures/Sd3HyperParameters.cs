using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// Hyper-parameters of an SD3-style MMDiT (joint-attention blocks over separate text and image streams, 2D sin-cos position
/// embeddings cropped from a fixed grid, optional per-head RMS norms and a second image-only attention in the first blocks, as in
/// SD3.5), read from <c>sd3.transformer.*</c>.
/// </summary>
internal sealed record Sd3HyperParameters
{
    /// <summary>Latent channels (before patching).</summary>
    public required int LatentChannels { get; init; }

    /// <summary>Latent pixels per token along each axis.</summary>
    public required int PatchSize { get; init; }

    /// <summary>Model width.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Attention heads.</summary>
    public required int HeadCount { get; init; }

    /// <summary>Joint-attention blocks (the last one only updates the image stream).</summary>
    public required int Layers { get; init; }

    /// <summary>Width of the text conditioning (the T5 hidden size; the CLIP states are zero-padded to it).</summary>
    public required int ContextDim { get; init; }

    /// <summary>Width of the pooled text embedding (CLIP-L + CLIP-G).</summary>
    public required int PooledDim { get; init; }

    /// <summary>Side of the square position-embedding grid the image's embeddings are cropped from (in tokens).</summary>
    public required int PositionGrid { get; init; }

    /// <summary>The first blocks with a second, image-only attention (SD3.5's MMDiT-X); 0 for SD3.</summary>
    public required int DualAttentionLayers { get; init; }

    /// <summary>Whether queries and keys are RMS-normed per head (SD3.5).</summary>
    public required bool QkNorm { get; init; }

    /// <summary>Native image size in pixels.</summary>
    public required int SampleSize { get; init; }

    /// <summary>Width per head.</summary>
    public int HeadDim => HiddenSize / HeadCount;

    /// <summary>Reads the keys.</summary>
    /// <param name="source">The canonical source.</param>
    /// <param name="prefix">The architecture prefix (<c>sd3.</c>).</param>
    public static Sd3HyperParameters Read(ModelSource source, string prefix)
    {
        string p = prefix + "transformer.";
        return new Sd3HyperParameters
        {
            LatentChannels = source.Get(p + "latent_channels", 16),
            PatchSize = source.Get(p + "patch_size", 2),
            HiddenSize = source.Get<int>(p + "hidden_size"),
            HeadCount = source.Get<int>(p + "head_count"),
            Layers = source.Get<int>(p + "layers"),
            ContextDim = source.Get<int>(p + "context_dim"),
            PooledDim = source.Get<int>(p + "pooled_dim"),
            PositionGrid = source.Get<int>(p + "position_grid"),
            DualAttentionLayers = source.Get(p + "dual_attention_layers", 0),
            QkNorm = source.Get(p + "qk_norm", false),
            SampleSize = source.Get(p + "sample_size", 1024),
        };
    }
}
