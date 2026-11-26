using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// The hyper-parameters of a ControlNet (canonical keys <c>controlnet.*</c>, weights <c>controlnet.*</c>): a copy of the UNet's time
/// embedding, input convolution, down path and middle block (<c>controlnet.unet.*</c>, the UNet keys; weights <c>controlnet.{input_conv,
/// time_embedding, class_embedding, down, mid}.*</c> named as the UNet's), the control image's embedding (<c>controlnet.condition_embedding.
/// {input_conv, blocks.{i}, output_conv}</c>) and the zero convolutions that turn its skip features into residuals of the UNet's
/// (<c>controlnet.zero_convs.{i}</c>, <c>controlnet.mid_zero_conv</c>).
/// </summary>
internal sealed record ControlNetHyperParameters
{
    /// <summary>The metadata prefix.</summary>
    public const string Prefix = "controlnet.";

    /// <summary>The canonical weight prefix.</summary>
    public const string WeightPrefix = "controlnet.";

    /// <summary>The copied UNet half.</summary>
    public required UNetHyperParameters UNet { get; init; }

    /// <summary>Channels of the control image (3).</summary>
    public required int ConditionChannels { get; init; }

    /// <summary>Channels of the control image's embedding levels (16, 32, 96, 256), each after the first at half the resolution.</summary>
    public required int[] ConditionEmbeddingChannels { get; init; }

    /// <summary>Whether a source holds a ControlNet.</summary>
    /// <param name="source">The canonical source.</param>
    public static bool IsPresent(ModelSource source) => source.Has(Prefix + "unet.in_channels");

    /// <summary>Reads the keys under <c>controlnet.</c>.</summary>
    /// <param name="source">The canonical source.</param>
    public static ControlNetHyperParameters Read(ModelSource source)
        => new()
        {
            UNet = UNetHyperParameters.Read(source, Prefix),
            ConditionChannels = source.Get(Prefix + "condition_channels", 3),
            ConditionEmbeddingChannels = source.Get<int[]>(Prefix + "condition_embedding_channels", [16, 32, 96, 256]),
        };
}
