namespace Sharplight.Audio;

/// <summary>Per-request settings of <see cref="IAudioEmbedder"/>.</summary>
public sealed record AudioEmbeddingOptions
{
    /// <summary>Whether the embedding is scaled to unit length (the form similarity search and zero-shot scoring use).</summary>
    public bool Normalize { get; init; } = true;

    /// <summary>What happens to audio longer than the model's clip (default: only the first clip).</summary>
    public LongAudioMode LongAudio { get; init; } = LongAudioMode.First;
}
