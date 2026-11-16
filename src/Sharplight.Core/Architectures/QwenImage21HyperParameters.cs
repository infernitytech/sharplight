using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// Hyper-parameters of a Qwen-Image 2.1 single-stream DiT (block-causal attention over [text, image], one shared modulation projection,
/// SwiGLU blocks, 3-axis RoPE), read from <c>qwen_image21.transformer.*</c>.
/// </summary>
internal sealed record QwenImage21HyperParameters
{
    /// <summary>Latent channels (one token per latent pixel: no patching).</summary>
    public required int LatentChannels { get; init; }

    /// <summary>Model width.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Attention heads.</summary>
    public required int HeadCount { get; init; }

    /// <summary>Blocks.</summary>
    public required int Layers { get; init; }

    /// <summary>Width of the text features.</summary>
    public required int ContextDim { get; init; }

    /// <summary>Rotated head dimensions per position axis (frame, row, column).</summary>
    public required int[] AxesDim { get; init; }

    /// <summary>RoPE base.</summary>
    public required float RopeTheta { get; init; }

    /// <summary>Norm epsilon.</summary>
    public required float NormEps { get; init; }

    /// <summary>Native image size in pixels.</summary>
    public required int SampleSize { get; init; }

    /// <summary>Whether text tokens are modulated from t = 0 (the released models' <c>causal_condition</c>).</summary>
    public required bool CausalCondition { get; init; }

    /// <summary>Width per head.</summary>
    public int HeadDim => HiddenSize / HeadCount;

    /// <summary>Reads the keys.</summary>
    /// <param name="source">The canonical source.</param>
    /// <param name="prefix">The architecture prefix (<c>qwen_image21.</c>).</param>
    public static QwenImage21HyperParameters Read(ModelSource source, string prefix)
    {
        string p = prefix + "transformer.";
        return new QwenImage21HyperParameters
        {
            LatentChannels = source.Get(p + "latent_channels", 64),
            HiddenSize = source.Get<int>(p + "hidden_size"),
            HeadCount = source.Get<int>(p + "head_count"),
            Layers = source.Get<int>(p + "layers"),
            ContextDim = source.Get<int>(p + "context_dim"),
            AxesDim = source.Get<int[]>(p + "axes_dim", [16, 56, 56]),
            RopeTheta = source.Get(p + "rope_theta", 10000f),
            NormEps = source.Get(p + "norm_eps", 1e-6f),
            SampleSize = source.Get(p + "sample_size", 1024),
            CausalCondition = source.Get(p + "causal_condition", true),
        };
    }
}
