using Sharplight.Audio;
using Sharplight.Llm;

namespace Sharplight.Speech;

/// <summary>Per-request settings of <see cref="ISpeechSynthesizer"/>; implementations ignore what their model cannot do.</summary>
public sealed record SpeechSynthesisOptions
{
    /// <summary>The voice (<see cref="ISpeechSynthesizer.Voices"/>), or <see langword="null"/> for the model's default voice.</summary>
    public string? Voice { get; init; }

    /// <summary>Language of the text (ISO 639-1, e.g. <c>en</c>), or <see langword="null"/> to let the model decide.</summary>
    public string? Language { get; init; }

    /// <summary>Speaking rate relative to the model's natural rate (1 = unchanged).</summary>
    public float Speed { get; init; } = 1f;

    /// <summary>
    /// Token selection of models that generate codec or acoustic tokens autoregressively. Left at its defaults (only <see cref="SamplingOptions.Seed"/>
    /// set, or nothing), the model's own sampling settings apply with that seed.
    /// </summary>
    public SamplingOptions Sampling { get; init; } = new();

    /// <summary>
    /// A recording of the voice to speak with (voice cloning), for models that take one (F5-TTS needs it); <see langword="null"/> for the
    /// <see cref="Voice"/>. Any rate and channel count.
    /// </summary>
    public AudioBuffer? ReferenceAudio { get; init; }

    /// <summary>The words spoken in <see cref="ReferenceAudio"/>, for models that align the reference with its text (F5-TTS).</summary>
    public string? ReferenceText { get; init; }

    /// <summary>The defaults.</summary>
    public static SpeechSynthesisOptions Default { get; } = new();
}
