using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>Hyper-parameters of a HunyuanVideo-style dual- and single-stream video transformer, read from <c>hunyuan_video.transformer.*</c>.</summary>
internal sealed record HunyuanVideoHyperParameters
{
    /// <summary>Latent channels of the input.</summary>
    public required int InChannels { get; init; }

    /// <summary>Latent channels of the prediction.</summary>
    public required int OutChannels { get; init; }

    /// <summary>Spatial patch size.</summary>
    public required int PatchSize { get; init; }

    /// <summary>Temporal patch size.</summary>
    public required int PatchSizeT { get; init; }

    /// <summary>Attention heads.</summary>
    public required int HeadCount { get; init; }

    /// <summary>Width per head.</summary>
    public required int HeadDim { get; init; }

    /// <summary>Dual-stream blocks.</summary>
    public required int DoubleLayers { get; init; }

    /// <summary>Single-stream blocks.</summary>
    public required int SingleLayers { get; init; }

    /// <summary>Blocks of the text token refiner.</summary>
    public required int RefinerLayers { get; init; }

    /// <summary>Width of the text states (the language model's hidden size).</summary>
    public required int TextDim { get; init; }

    /// <summary>Width of the pooled text embedding (CLIP-L).</summary>
    public required int PooledDim { get; init; }

    /// <summary>Whether the guidance scale is embedded (guidance-distilled).</summary>
    public required bool GuidanceEmbed { get; init; }

    /// <summary>RoPE base.</summary>
    public required float RopeTheta { get; init; }

    /// <summary>Rotated head dimensions per axis (frames, rows, columns).</summary>
    public required int[] AxesDim { get; init; }

    /// <summary>Model width.</summary>
    public int HiddenSize => HeadCount * HeadDim;

    /// <summary>Reads the keys.</summary>
    /// <param name="source">The canonical source.</param>
    /// <param name="prefix">The architecture prefix (<c>hunyuan_video.</c>).</param>
    public static HunyuanVideoHyperParameters Read(ModelSource source, string prefix)
    {
        string p = prefix + "transformer.";
        int inChannels = source.Get<int>(p + "in_channels");
        return new HunyuanVideoHyperParameters
        {
            InChannels = inChannels,
            OutChannels = source.Get(p + "out_channels", inChannels),
            PatchSize = source.Get(p + "patch_size", 2),
            PatchSizeT = source.Get(p + "patch_size_t", 1),
            HeadCount = source.Get<int>(p + "head_count"),
            HeadDim = source.Get(p + "head_dim", 128),
            DoubleLayers = source.Get<int>(p + "double_layers"),
            SingleLayers = source.Get<int>(p + "single_layers"),
            RefinerLayers = source.Get(p + "refiner_layers", 2),
            TextDim = source.Get<int>(p + "text_dim"),
            PooledDim = source.Get(p + "pooled_dim", 768),
            GuidanceEmbed = source.Get(p + "guidance_embed", true),
            RopeTheta = source.Get(p + "rope_theta", 256f),
            AxesDim = source.Get<int[]>(p + "rope_axes_dim", [16, 56, 56]),
        };
    }
}
