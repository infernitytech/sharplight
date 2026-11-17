using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>Hyper-parameters of an LTX-Video diffusion transformer, read from <c>ltx.transformer.*</c>.</summary>
internal sealed record LtxHyperParameters
{
    /// <summary>Latent channels of the input and the prediction.</summary>
    public required int LatentChannels { get; init; }

    /// <summary>Model width.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Attention heads.</summary>
    public required int HeadCount { get; init; }

    /// <summary>Transformer blocks.</summary>
    public required int LayerCount { get; init; }

    /// <summary>Width of the text conditioning (the T5 hidden size).</summary>
    public required int CaptionDim { get; init; }

    /// <summary>Epsilon of the block norms.</summary>
    public required float NormEps { get; init; }

    /// <summary>RoPE base.</summary>
    public required float RopeTheta { get; init; }

    /// <summary>RoPE grid normalization: frames, rows, columns of the base grid.</summary>
    public required float[] RopeBase { get; init; }

    /// <summary>
    /// Whether the RoPE frame coordinate of a latent frame is its last pixel frame, max(f · t + 1 − t, 0), as Lightricks' inference and
    /// diffusers' condition pipeline compute it, instead of f · t (diffusers' <c>LTXPipeline</c>).
    /// </summary>
    public bool CausalFrameCoordinates { get; init; }

    /// <summary>Width per head.</summary>
    public int HeadDim => HiddenSize / HeadCount;

    /// <summary>Reads the keys.</summary>
    /// <param name="source">The canonical source.</param>
    /// <param name="prefix">The architecture prefix (<c>ltx.</c>).</param>
    public static LtxHyperParameters Read(ModelSource source, string prefix)
    {
        string p = prefix + "transformer.";
        return new LtxHyperParameters
        {
            LatentChannels = source.Get<int>(p + "latent_channels"),
            HiddenSize = source.Get<int>(p + "hidden_size"),
            HeadCount = source.Get<int>(p + "head_count"),
            LayerCount = source.Get<int>(p + "layer_count"),
            CaptionDim = source.Get<int>(p + "caption_dim"),
            NormEps = source.Get(p + "norm_eps", 1e-6f),
            RopeTheta = source.Get(p + "rope_theta", 10000f),
            RopeBase = source.Get<float[]>(p + "rope_base", [20f, 2048f, 2048f]),
            CausalFrameCoordinates = source.Get(p + "causal_frame_coordinates", false),
        };
    }
}
