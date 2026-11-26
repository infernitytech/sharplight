namespace Sharplight.VideoEnhancement;

/// <summary>
/// Per-request settings of <see cref="IVideoEnhancer"/>; <see langword="null"/> values take the model's defaults
/// (<see cref="VideoEnhancementDefaults"/>).
/// </summary>
public sealed record VideoEnhancementOptions
{
    /// <summary>
    /// Output width in pixels, or <see langword="null"/>. With only one of <see cref="Width"/> and <see cref="Height"/> the other follows the
    /// input's aspect ratio; with neither, <see cref="Scale"/> applies. Sizes the network does not take are padded and cropped back, so the
    /// result has exactly this size.
    /// </summary>
    public int? Width { get; init; }

    /// <summary>Output height in pixels, or <see langword="null"/> (see <see cref="Width"/>).</summary>
    public int? Height { get; init; }

    /// <summary>The enlargement of both sides when no size is given, or <see langword="null"/> for the model's default.</summary>
    public float? Scale { get; init; }

    /// <summary>
    /// Frames per batch, or <see langword="null"/> for the model's default (0: the whole clip in one batch, up to the model's
    /// <see cref="VideoEnhancementDefaults.MaxFramesPerBatch"/>). Each batch is restored on its own, so longer clips fit in memory; a value
    /// the model does not take is rounded down to one it does (1 + k · the frame multiple).
    /// </summary>
    public int? FramesPerBatch { get; init; }

    /// <summary>
    /// The longest side, in pixels of the output, of the tiles the model's autoencoder runs on (sizes whose activations do not fit the device
    /// are encoded and decoded in overlapping tiles that are crossfaded together), or <see langword="null"/> to choose the tiles from the
    /// device's free memory: none while the whole frame fits, smaller ones after the device runs out of memory. Smaller tiles need less
    /// memory and take a little longer; the seams are blended, but the result differs slightly from an untiled run.
    /// </summary>
    public int? TileSize { get; init; }

    /// <summary>How the colours of the result follow the input, or <see langword="null"/> for the model's default.</summary>
    public VideoColorCorrection? ColorCorrection { get; init; }

    /// <summary>Seed of the model's noise; <see langword="null"/> picks one (reported in <see cref="VideoEnhancementResult.Seed"/>).</summary>
    public long? Seed { get; init; }

    /// <summary>
    /// Explicit noise the restoration starts from, instead of the seeded noise: unit Gaussian values in the model's latent layout
    /// [channels, latent frames, height / s, width / s] of the whole clip as one batch; reproduces another implementation's run exactly.
    /// </summary>
    public float[]? InitialNoise { get; init; }

    /// <summary>
    /// Explicit unit Gaussian values for sampling the encoded input's latent distribution (models whose encoder is a variational one), in the
    /// layout of <see cref="InitialNoise"/>; reproduces another implementation's run exactly.
    /// </summary>
    public float[]? EncoderNoise { get; init; }

    /// <summary>The defaults.</summary>
    public static VideoEnhancementOptions Default { get; } = new();
}
