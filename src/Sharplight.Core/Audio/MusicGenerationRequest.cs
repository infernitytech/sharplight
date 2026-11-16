namespace Sharplight.Audio;

/// <summary>One request to <see cref="IMusicGenerator"/>.</summary>
public sealed record MusicGenerationRequest
{
    /// <summary>The description of the music (genre, instruments, mood).</summary>
    public required string Prompt { get; init; }

    /// <summary>The duration of the music (default 10 s; limited by the model's context, 30 s for MusicGen).</summary>
    public TimeSpan Duration { get; init; } = TimeSpan.FromSeconds(10);

    /// <summary>Classifier-free guidance scale (1 disables guidance), or <see langword="null"/> for the model's default (3).</summary>
    public float? GuidanceScale { get; init; }

    /// <summary>True for greedy decoding (deterministic; the form that is checked token by token against the reference), false to sample.</summary>
    public bool Greedy { get; init; }

    /// <summary>The sampling temperature.</summary>
    public float Temperature { get; init; } = 1f;

    /// <summary>The sampling keeps the best <c>TopK</c> codes (0 for all of them).</summary>
    public int TopK { get; init; } = 250;

    /// <summary>The sampling keeps the smallest set of codes whose probability reaches <c>TopP</c> (1 for all of them).</summary>
    public float TopP { get; init; } = 1f;

    /// <summary>The seed of the sampling, or <see langword="null"/> for a random one.</summary>
    public int? Seed { get; init; }
}
