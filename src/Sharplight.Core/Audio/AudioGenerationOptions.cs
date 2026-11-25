namespace Sharplight.Audio;

/// <summary>Per-request settings of <see cref="IAudioGenerator"/>; <see langword="null"/> values take the model's defaults (<see cref="IAudioGenerator.Defaults"/>).</summary>
public sealed record AudioGenerationOptions
{
    /// <summary>
    /// The number of denoising steps, or <see langword="null"/> for the model's default (<see cref="AudioGenerationDefaults.Steps"/>: 8 for the distilled ACE-Step model,
    /// 100 for Stable Audio Open, 50 for the Stable Audio 3 base models).
    /// </summary>
    public int? Steps { get; init; }

    /// <summary>
    /// Classifier-free guidance scale (1 disables guidance), or <see langword="null"/> for the model's default (<see cref="AudioGenerationDefaults.GuidanceScale"/>).
    /// Distilled models that were trained without guidance ignore it.
    /// </summary>
    public float? GuidanceScale { get; init; }

    /// <summary>
    /// The seed of the initial noise; <see langword="null"/> picks one (reported in <see cref="AudioGenerationResult.Seed"/>). Seeds within the <see cref="int"/> range
    /// reproduce the runs of earlier releases.
    /// </summary>
    public long? Seed { get; init; }

    /// <summary>
    /// The initial noise itself, <c>frames · channels</c> values in frame-major order (<see cref="IAudioGenerator.LatentFrameRate"/> frames per second,
    /// <see cref="IAudioGenerator.LatentChannels"/> channels), for reproducing another implementation's run; replaces <see cref="Seed"/>.
    /// </summary>
    public ReadOnlyMemory<float>? InitialNoise { get; init; }

    /// <summary>
    /// The noise of the stochastic sampler's steps, one array of <c>frames · channels</c> values (frame-major, as <see cref="InitialNoise"/>) per step, for reproducing
    /// another implementation's run of a model with a stochastic sampler (Stable Audio's DPM-Solver++ SDE); <see langword="null"/> draws it from the seed. Models with
    /// a deterministic sampler ignore it.
    /// </summary>
    public IReadOnlyList<ReadOnlyMemory<float>>? StepNoise { get; init; }

    /// <summary>
    /// Scales the random noise that some autoencoders add while they decode (Stable Audio 3's SAME decoder dithers its latents and the tokens it generates), or
    /// <see langword="null"/> for the model's own amount; 0 decodes deterministically. Drawn from <see cref="Seed"/>. Other models ignore it.
    /// </summary>
    public float? DecoderNoise { get; init; }

    /// <summary>
    /// The shift of the timestep schedule (larger values spend more steps at high noise), or <see langword="null"/> for the model's default; for models with
    /// <see cref="AudioGenerationDefaults.SupportsShift"/>, others ignore it.
    /// </summary>
    public float? Shift { get; init; }

    /// <summary>The defaults.</summary>
    public static AudioGenerationOptions Default { get; } = new();
}
