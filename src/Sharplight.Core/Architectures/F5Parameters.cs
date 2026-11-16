using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// The canonical hyper-parameters of F5-TTS (<c>f5tts.*</c>): the transformer width, depth and heads, the text embedding width and ConvNeXt block
/// count, the mel bins, the time-sinusoid width, the convolutional position embedding's kernel and groups, and the sampling defaults (Euler steps,
/// guidance strength, sway coefficient).
/// </summary>
internal sealed record F5Parameters
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "f5tts";

    /// <summary>Transformer width.</summary>
    public required int Dim { get; init; }

    /// <summary>DiT blocks.</summary>
    public required int Depth { get; init; }

    /// <summary>Attention heads.</summary>
    public required int Heads { get; init; }

    /// <summary>Text embedding width.</summary>
    public required int TextDim { get; init; }

    /// <summary>ConvNeXt V2 blocks of the text embedding.</summary>
    public required int TextBlocks { get; init; }

    /// <summary>Mel bins.</summary>
    public required int MelBins { get; init; }

    /// <summary>Width of the time step's sinusoid.</summary>
    public int TimeChannels { get; init; } = 256;

    /// <summary>Kernel of the convolutional position embedding.</summary>
    public int ConvPositionKernel { get; init; } = 31;

    /// <summary>Groups of the convolutional position embedding.</summary>
    public int ConvPositionGroups { get; init; } = 16;

    /// <summary>Euler steps of the sampler.</summary>
    public int Steps { get; init; } = 32;

    /// <summary>Classifier-free guidance strength.</summary>
    public float Guidance { get; init; } = 2f;

    /// <summary>Sway sampling coefficient.</summary>
    public float Sway { get; init; } = -1f;

    /// <summary>Reads the parameters of a source.</summary>
    /// <param name="src">The model source.</param>
    public static F5Parameters From(ModelSource src) => new()
    {
        Dim = src.Arch<int>("dim"),
        Depth = src.Arch<int>("depth"),
        Heads = src.Arch<int>("heads"),
        TextDim = src.Arch<int>("text_dim"),
        TextBlocks = src.Arch<int>("text_blocks"),
        MelBins = src.Arch<int>("mel_bins"),
        TimeChannels = src.Arch("time_channels", 256),
        ConvPositionKernel = src.Arch("conv_position_kernel", 31),
        ConvPositionGroups = src.Arch("conv_position_groups", 16),
        Steps = src.Arch("steps", 32),
        Guidance = src.Arch("guidance", 2f),
        Sway = src.Arch("sway", -1f),
    };
}
