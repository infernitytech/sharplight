using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>Hyper-parameters of a Wan-style video diffusion transformer, read from <c>wan.transformer.*</c>.</summary>
internal sealed record WanHyperParameters
{
    /// <summary>Latent channels of the input (before patching).</summary>
    public required int InChannels { get; init; }

    /// <summary>Latent channels of the prediction.</summary>
    public required int OutChannels { get; init; }

    /// <summary>Patch size along (frames, rows, columns).</summary>
    public required int[] PatchSize { get; init; }

    /// <summary>Model width.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Attention heads.</summary>
    public required int HeadCount { get; init; }

    /// <summary>Transformer blocks.</summary>
    public required int LayerCount { get; init; }

    /// <summary>Inner width of the feed-forward layers.</summary>
    public required int FfnDim { get; init; }

    /// <summary>Width of the text conditioning (the umT5 hidden size).</summary>
    public required int TextDim { get; init; }

    /// <summary>Channels of the sinusoidal timestep embedding.</summary>
    public required int FrequencyDim { get; init; }

    /// <summary>Epsilon of the layer and RMS norms.</summary>
    public required float NormEps { get; init; }

    /// <summary>Whether the cross-attention input is layer-normed (with affine parameters).</summary>
    public required bool CrossAttentionNorm { get; init; }

    /// <summary>RoPE base.</summary>
    public required float RopeTheta { get; init; }

    /// <summary>Whether the timestep is an input per token (Wan 2.2 TI2V).</summary>
    public required bool TokenTimesteps { get; init; }

    /// <summary>Width per head.</summary>
    public int HeadDim => HiddenSize / HeadCount;

    /// <summary>Rotated head dimensions per axis (frames, rows, columns): rows and columns 2·⌊d/6⌋ each, frames the rest.</summary>
    public int[] AxesDim
    {
        get
        {
            int spatial = 2 * (HeadDim / 6);
            return [HeadDim - 2 * spatial, spatial, spatial];
        }
    }

    /// <summary>Reads the keys.</summary>
    /// <param name="source">The canonical source.</param>
    /// <param name="prefix">The architecture prefix (<c>wan.</c>).</param>
    public static WanHyperParameters Read(ModelSource source, string prefix)
    {
        string p = prefix + "transformer.";
        int inChannels = source.Get<int>(p + "in_channels");
        return new WanHyperParameters
        {
            InChannels = inChannels,
            OutChannels = source.Get(p + "out_channels", inChannels),
            PatchSize = source.Get<int[]>(p + "patch_size", [1, 2, 2]),
            HiddenSize = source.Get<int>(p + "hidden_size"),
            HeadCount = source.Get<int>(p + "head_count"),
            LayerCount = source.Get<int>(p + "layer_count"),
            FfnDim = source.Get<int>(p + "ffn_dim"),
            TextDim = source.Get<int>(p + "text_dim"),
            FrequencyDim = source.Get(p + "freq_dim", 256),
            NormEps = source.Get(p + "norm_eps", 1e-6f),
            CrossAttentionNorm = source.Get(p + "cross_attn_norm", true),
            RopeTheta = source.Get(p + "rope_theta", 10000f),
            TokenTimesteps = source.Get(p + "token_timesteps", false),
        };
    }
}
