using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// The hyper-parameters of a tiny autoencoder (TAESD; canonical keys <c>tiny_autoencoder.*</c>, weights <c>tiny_vae.*</c>): plain
/// convolution stacks of ReLU residual blocks, four levels of <c>channels</c> each, nearest upsampling in the decoder and strided
/// convolutions in the encoder.
/// </summary>
internal sealed record TinyAutoencoderHyperParameters
{
    /// <summary>The metadata prefix.</summary>
    public const string Prefix = "tiny_autoencoder.";

    /// <summary>The canonical weight prefix.</summary>
    public const string WeightPrefix = "tiny_vae.";

    /// <summary>Latent channels (TAESD / TAESDXL 4, TAESD3 / TAEF1 16).</summary>
    public required int LatentChannels { get; init; }

    /// <summary>Residual blocks per decoder level, coarsest first.</summary>
    public required int[] DecoderBlocks { get; init; }

    /// <summary>Residual blocks per encoder level, finest first.</summary>
    public required int[] EncoderBlocks { get; init; }

    /// <summary>Decoder channels per level.</summary>
    public required int[] DecoderChannels { get; init; }

    /// <summary>Encoder channels per level.</summary>
    public required int[] EncoderChannels { get; init; }

    /// <summary>The decoder's upsampling factor per level.</summary>
    public required int UpsampleFactor { get; init; }

    /// <summary>The autoencoder's latents are <c>(model latents) / scale + shift</c>; TAESD's are the model latents themselves (1, 0).</summary>
    public required float ScalingFactor { get; init; }

    /// <summary>See <see cref="ScalingFactor"/>.</summary>
    public required float ShiftFactor { get; init; }

    /// <summary>Whether the decoder's weights are there.</summary>
    public required bool HasDecoder { get; init; }

    /// <summary>Whether the encoder's weights are there.</summary>
    public required bool HasEncoder { get; init; }

    /// <summary>Pixels per latent along each axis.</summary>
    public int Downscale => (int)Math.Pow(UpsampleFactor, DecoderBlocks.Length - 1);

    /// <summary>Whether a source holds a tiny autoencoder.</summary>
    /// <param name="source">The canonical source.</param>
    public static bool IsPresent(ModelSource source) => source.Has(Prefix + "latent_channels");

    /// <summary>Reads the keys under <c>tiny_autoencoder.</c>.</summary>
    /// <param name="source">The canonical source.</param>
    public static TinyAutoencoderHyperParameters Read(ModelSource source)
        => new()
        {
            LatentChannels = source.Get<int>(Prefix + "latent_channels"),
            DecoderBlocks = source.Get<int[]>(Prefix + "decoder_blocks", [3, 3, 3, 1]),
            EncoderBlocks = source.Get<int[]>(Prefix + "encoder_blocks", [1, 3, 3, 3]),
            DecoderChannels = source.Get<int[]>(Prefix + "decoder_channels", [64, 64, 64, 64]),
            EncoderChannels = source.Get<int[]>(Prefix + "encoder_channels", [64, 64, 64, 64]),
            UpsampleFactor = source.Get(Prefix + "upsample_factor", 2),
            ScalingFactor = source.Get(Prefix + "scaling_factor", 1f),
            ShiftFactor = source.Get(Prefix + "shift_factor", 0f),
            HasDecoder = source.Get(Prefix + "has_decoder", true),
            HasEncoder = source.Get(Prefix + "has_encoder", false),
        };
}
