using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// Hyper-parameters of a Z-Image single-stream DiT (refiner blocks over the image and the caption tokens separately, then joint blocks
/// over both; tanh-gated, sandwich-normed RMS blocks with a SwiGLU feed-forward; 3-axis RoPE), read from <c>zimage.transformer.*</c>.
/// </summary>
internal sealed record ZImageHyperParameters
{
    /// <summary>Latent channels (before patching).</summary>
    public required int LatentChannels { get; init; }

    /// <summary>Latent pixels per token along each axis.</summary>
    public required int PatchSize { get; init; }

    /// <summary>Model width.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Attention heads.</summary>
    public required int HeadCount { get; init; }

    /// <summary>Joint blocks.</summary>
    public required int Layers { get; init; }

    /// <summary>Refiner blocks of each stream (image: modulated; caption: not).</summary>
    public required int RefinerLayers { get; init; }

    /// <summary>Width of the caption features (the text encoder's hidden size).</summary>
    public required int CaptionDim { get; init; }

    /// <summary>Width of the timestep embedding the modulations are projected from.</summary>
    public required int ModulationDim { get; init; }

    /// <summary>Rotated head dimensions per position axis (sequence index, row, column).</summary>
    public required int[] AxesDim { get; init; }

    /// <summary>RoPE base.</summary>
    public required float RopeTheta { get; init; }

    /// <summary>Scale of the timestep (1 − σ) before its sinusoidal embedding.</summary>
    public required float TimeScale { get; init; }

    /// <summary>RMS norm epsilon.</summary>
    public required float NormEps { get; init; }

    /// <summary>Native image size in pixels.</summary>
    public required int SampleSize { get; init; }

    /// <summary>Width per head.</summary>
    public int HeadDim => HiddenSize / HeadCount;

    /// <summary>Reads the keys.</summary>
    /// <param name="source">The canonical source.</param>
    /// <param name="prefix">The architecture prefix (<c>zimage.</c>).</param>
    public static ZImageHyperParameters Read(ModelSource source, string prefix)
    {
        string p = prefix + "transformer.";
        return new ZImageHyperParameters
        {
            LatentChannels = source.Get(p + "latent_channels", 16),
            PatchSize = source.Get(p + "patch_size", 2),
            HiddenSize = source.Get<int>(p + "hidden_size"),
            HeadCount = source.Get<int>(p + "head_count"),
            Layers = source.Get<int>(p + "layers"),
            RefinerLayers = source.Get(p + "refiner_layers", 2),
            CaptionDim = source.Get<int>(p + "caption_dim"),
            ModulationDim = source.Get(p + "modulation_dim", 256),
            AxesDim = source.Get<int[]>(p + "axes_dim", [32, 48, 48]),
            RopeTheta = source.Get(p + "rope_theta", 256f),
            TimeScale = source.Get(p + "time_scale", 1000f),
            NormEps = source.Get(p + "norm_eps", 1e-5f),
            SampleSize = source.Get(p + "sample_size", 1024),
        };
    }
}
