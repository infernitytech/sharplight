namespace Sharplight.Audio;

/// <summary>What to generate (<see cref="IAudioGenerator"/>); how (steps, guidance, seed) is in <see cref="AudioGenerationOptions"/>.</summary>
public sealed record AudioGenerationRequest
{
    /// <summary>The description of the audio (genre, instruments, mood, a sound effect).</summary>
    public required string Prompt { get; init; }

    /// <summary>
    /// What the audio should not sound like (the unconditional side of classifier-free guidance), for models with <see cref="AudioGenerationDefaults.SupportsNegativePrompt"/>;
    /// <see langword="null"/> guides away from the model's empty conditioning.
    /// </summary>
    public string? NegativePrompt { get; init; }

    /// <summary>
    /// The lyrics of a song with the section tags of the model (<c>[verse]</c>, <c>[chorus]</c>), or <c>[Instrumental]</c> / <see langword="null"/> for none; for
    /// models with <see cref="AudioGenerationDefaults.SupportsLyrics"/>.
    /// </summary>
    public string? Lyrics { get; init; }

    /// <summary>The language of the lyrics (a code such as <c>en</c>, <c>zh</c>, <c>ja</c>), or <see langword="null"/> for <c>unknown</c>.</summary>
    public string? Language { get; init; }

    /// <summary>The duration of the audio (default 10 s; limited by <see cref="IAudioGenerator.MaxDuration"/>).</summary>
    public TimeSpan Duration { get; init; } = TimeSpan.FromSeconds(10);
}
