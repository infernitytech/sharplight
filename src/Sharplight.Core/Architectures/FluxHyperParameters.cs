using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// Hyper-parameters of a FLUX-style rectified-flow transformer (MMDiT: dual-stream blocks over separate text and image streams, then
/// single-stream blocks over the joined sequence), read from <c>flux.transformer.*</c>.
/// </summary>
internal sealed record FluxHyperParameters
{
    /// <summary>Latent channels (before patching).</summary>
    public required int LatentChannels { get; init; }

    /// <summary>Latent pixels per token along each axis.</summary>
    public required int PatchSize { get; init; }

    /// <summary>Model width.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Attention heads.</summary>
    public required int HeadCount { get; init; }

    /// <summary>Dual-stream blocks.</summary>
    public required int DoubleLayers { get; init; }

    /// <summary>Single-stream blocks.</summary>
    public required int SingleLayers { get; init; }

    /// <summary>Width of the text conditioning (the T5 hidden size).</summary>
    public required int ContextDim { get; init; }

    /// <summary>Width of the pooled text embedding (CLIP).</summary>
    public required int PooledDim { get; init; }

    /// <summary>Whether the guidance scale is embedded (guidance-distilled models, FLUX.1-dev).</summary>
    public required bool GuidanceEmbed { get; init; }

    /// <summary>Rotated head dimensions per position axis (text/index, row, column); they sum to the head width.</summary>
    public required int[] AxesDim { get; init; }

    /// <summary>RoPE base.</summary>
    public required float RopeTheta { get; init; }

    /// <summary>Native image size in pixels.</summary>
    public required int SampleSize { get; init; }

    /// <summary>Width per head.</summary>
    public int HeadDim => HiddenSize / HeadCount;

    /// <summary>Reads the keys.</summary>
    /// <param name="source">The canonical source.</param>
    /// <param name="prefix">The architecture prefix (<c>flux.</c>).</param>
    public static FluxHyperParameters Read(ModelSource source, string prefix)
    {
        string p = prefix + "transformer.";
        return new FluxHyperParameters
        {
            LatentChannels = source.Get(p + "latent_channels", 16),
            PatchSize = source.Get(p + "patch_size", 2),
            HiddenSize = source.Get<int>(p + "hidden_size"),
            HeadCount = source.Get<int>(p + "head_count"),
            DoubleLayers = source.Get<int>(p + "double_layers"),
            SingleLayers = source.Get<int>(p + "single_layers"),
            ContextDim = source.Get<int>(p + "context_dim"),
            PooledDim = source.Get<int>(p + "pooled_dim"),
            GuidanceEmbed = source.Get(p + "guidance_embed", false),
            AxesDim = source.Get<int[]>(p + "axes_dim", [16, 56, 56]),
            RopeTheta = source.Get(p + "rope_theta", 10000f),
            SampleSize = source.Get(p + "sample_size", 1024),
        };
    }
}
