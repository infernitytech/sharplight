namespace Sharplight.Audio;

/// <summary>Settings of one <see cref="IVoiceActivityDetector"/> request: how the per-frame speech probabilities become segments.</summary>
public sealed record VoiceActivityOptions
{
    /// <summary>A frame is speech when its probability is at least this. Default 0.5.</summary>
    public float Threshold { get; init; } = 0.5f;

    /// <summary>Inside speech, a frame is silence when its probability is below this; <see langword="null"/> for <c>max(Threshold - 0.15, 0.01)</c>.</summary>
    public float? NegativeThreshold { get; init; }

    /// <summary>Segments shorter than this are dropped. Default 250 ms.</summary>
    public TimeSpan MinSpeechDuration { get; init; } = TimeSpan.FromMilliseconds(250);

    /// <summary>Segments longer than this are split, at the longest silence inside them if there is one. Default: unlimited.</summary>
    public TimeSpan MaxSpeechDuration { get; init; } = TimeSpan.MaxValue;

    /// <summary>Silence this long ends a segment. Default 100 ms.</summary>
    public TimeSpan MinSilenceDuration { get; init; } = TimeSpan.FromMilliseconds(100);

    /// <summary>Every segment is extended by this much on both sides (neighbours meet in the middle of their gap). Default 30 ms.</summary>
    public TimeSpan SpeechPad { get; init; } = TimeSpan.FromMilliseconds(30);

    /// <summary>The shortest silence that counts as a place to split an overlong segment. Default 98 ms.</summary>
    public TimeSpan MinSilenceAtMaxSpeech { get; init; } = TimeSpan.FromMilliseconds(98);

    /// <summary>Whether an overlong segment is split at its longest silence (true) or at its last one (false). Default true.</summary>
    public bool UseLongestSilenceAtMaxSpeech { get; init; } = true;
}
