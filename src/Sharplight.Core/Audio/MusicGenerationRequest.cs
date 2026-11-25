namespace Sharplight.Audio;

/// <summary>What to generate (<see cref="IMusicGenerator"/>); how (guidance, sampling, seed) is in <see cref="MusicGenerationOptions"/>.</summary>
public sealed record MusicGenerationRequest
{
    /// <summary>The description of the music (genre, instruments, mood).</summary>
    public required string Prompt { get; init; }

    /// <summary>The duration of the music (default 10 s; limited by the model's context, 30 s for MusicGen).</summary>
    public TimeSpan Duration { get; init; } = TimeSpan.FromSeconds(10);
}
