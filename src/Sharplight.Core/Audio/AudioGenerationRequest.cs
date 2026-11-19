namespace Sharplight.Audio;

/// <summary>One request to <see cref="IAudioGenerator"/>.</summary>
public sealed record AudioGenerationRequest
{
    /// <summary>The description of the audio (genre, instruments, mood, a sound effect).</summary>
    public required string Prompt { get; init; }

    /// <summary>
    /// What the audio should not sound like (the unconditional side of classifier-free guidance), for models with <see cref="IAudioGenerator.SupportsNegativePrompt"/>;
    /// <see langword="null"/> guides away from the model's empty conditioning.
    /// </summary>
    public string? NegativePrompt { get; init; }

    /// <summary>The lyrics of a song with the section tags of the model (<c>[verse]</c>, <c>[chorus]</c>), or <c>[Instrumental]</c> / <see langword="null"/> for none.</summary>
    public string? Lyrics { get; init; }

    /// <summary>The language of the lyrics (a code such as <c>en</c>, <c>zh</c>, <c>ja</c>), or <see langword="null"/> for <c>unknown</c>.</summary>
    public string? Language { get; init; }

    /// <summary>The duration of the audio (default 10 s; limited by <see cref="IAudioGenerator.MaxDuration"/>).</summary>
    public TimeSpan Duration { get; init; } = TimeSpan.FromSeconds(10);

    /// <summary>The number of denoising steps, or <see langword="null"/> for the model's default (<see cref="IAudioGenerator.DefaultSteps"/>: 8 for the distilled ACE-Step model, 100 for Stable Audio Open, 50 for the Stable Audio 3 base models).</summary>
    public int? Steps { get; init; }

    /// <summary>
    /// Classifier-free guidance scale (1 disables guidance), or <see langword="null"/> for the model's default (<see cref="IAudioGenerator.DefaultGuidanceScale"/>).
    /// Distilled models that were trained without guidance ignore it.
    /// </summary>
    public float? GuidanceScale { get; init; }

    /// <summary>The seed of the initial noise, or <see langword="null"/> for a random one.</summary>
    public int? Seed { get; init; }

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

    /// <summary>The shift of the timestep schedule (larger values spend more steps at high noise), or <see langword="null"/> for the model's default.</summary>
    public float? Shift { get; init; }
}
