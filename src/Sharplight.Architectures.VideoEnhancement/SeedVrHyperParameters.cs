using Sharplight.Models;

namespace Sharplight.Architectures.VideoEnhancement;

/// <summary>Hyper-parameters of a SeedVR2 restoration DiT, read from <c>seedvr.transformer.*</c>.</summary>
internal sealed record SeedVrHyperParameters
{
    /// <summary>Model width of both streams.</summary>
    public required int Dim { get; init; }

    /// <summary>Attention heads.</summary>
    public required int Heads { get; init; }

    /// <summary>Width per head.</summary>
    public required int HeadDim { get; init; }

    /// <summary>Transformer blocks.</summary>
    public required int Layers { get; init; }

    /// <summary>Width of the text embeddings.</summary>
    public required int TextDim { get; init; }

    /// <summary>Latent channels of the input before patching (noise, condition and the condition mask).</summary>
    public required int InChannels { get; init; }

    /// <summary>Latent channels of the prediction.</summary>
    public required int OutChannels { get; init; }

    /// <summary>Patch size along (frames, rows, columns).</summary>
    public required int[] PatchSize { get; init; }

    /// <summary>Epsilon of the RMS norms.</summary>
    public required float NormEps { get; init; }

    /// <summary>Channels of the sinusoidal timestep embedding.</summary>
    public required int TimeEmbedDim { get; init; }

    /// <summary>Whether the feed-forward is SwiGLU (<c>gate</c>, <c>up</c>, <c>down</c> without biases) rather than a tanh-GELU MLP.</summary>
    public required bool GatedMlp { get; init; }

    /// <summary>
    /// The RoPE positions: <see langword="true"/> for integer positions over the text too, the video's frame positions offset by the text
    /// length (3B); <see langword="false"/> for video-only positions spread evenly over [−1, 1] across each window (7B).
    /// </summary>
    public required bool TextRope { get; init; }

    /// <summary>
    /// Whether the last block passes the text stream through unmodulated (its attention still reads the text's keys and values) and skips
    /// the text feed-forward (3B).
    /// </summary>
    public required bool LastLayerTextPassthrough { get; init; }

    /// <summary>Windows per axis (frames, rows, columns) on the reference grid.</summary>
    public required int[] WindowCounts { get; init; }

    /// <summary>The token grid (rows, columns) the window counts refer to (720p).</summary>
    public required int[] WindowReferenceGrid { get; init; }

    /// <summary>The largest frame count the temporal window count refers to.</summary>
    public required int WindowMaxFrames { get; init; }

    /// <summary>Reads the hyper-parameters.</summary>
    /// <param name="src">The canonical source.</param>
    /// <param name="prefix">The architecture prefix (<c>seedvr.</c>).</param>
    public static SeedVrHyperParameters Read(ModelSource src, string prefix)
    {
        string p = prefix + "transformer.";
        return new SeedVrHyperParameters
        {
            Dim = src.Get<int>(p + "dim"),
            Heads = src.Get<int>(p + "heads"),
            HeadDim = src.Get<int>(p + "head_dim"),
            Layers = src.Get<int>(p + "layers"),
            TextDim = src.Get<int>(p + "text_dim"),
            InChannels = src.Get<int>(p + "in_channels"),
            OutChannels = src.Get<int>(p + "out_channels"),
            PatchSize = src.Get<int[]>(p + "patch_size"),
            NormEps = src.Get(p + "norm_eps", 1e-5f),
            TimeEmbedDim = src.Get(p + "time_embed_dim", 256),
            GatedMlp = src.Get(p + "mlp", "swiglu") == "swiglu",
            TextRope = src.Get(p + "rope", "text_offset") == "text_offset",
            LastLayerTextPassthrough = src.Get(p + "last_layer_text_passthrough", false),
            WindowCounts = src.Get<int[]>(p + "window_counts"),
            WindowReferenceGrid = src.Get<int[]>(p + "window_reference_grid"),
            WindowMaxFrames = src.Get(p + "window_max_frames", 30),
        };
    }
}
