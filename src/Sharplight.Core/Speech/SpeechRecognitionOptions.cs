using Sharplight.Audio;
using Sharplight.Llm;

namespace Sharplight.Speech;

/// <summary>Per-request settings of <see cref="ISpeechRecognizer"/>; implementations ignore what their model cannot do.</summary>
public sealed record SpeechRecognitionOptions
{
    /// <summary>Spoken language (ISO 639-1, e.g. <c>en</c>), or <see langword="null"/> to detect it.</summary>
    public string? Language { get; init; }

    /// <summary>Translate the speech into English instead of transcribing it (multilingual Whisper-style models).</summary>
    public bool Translate { get; init; }

    /// <summary>Timestamp detail of the result.</summary>
    public TimestampGranularity Timestamps { get; init; } = TimestampGranularity.Segment;

    /// <summary>Text that conditions the decoder (vocabulary, spelling, style), or <see langword="null"/>.</summary>
    public string? Prompt { get; init; }

    /// <summary>
    /// Long recordings: prompt each window with the text of the previous ones (more consistent spelling and style, but a misrecognized window
    /// can derail the next ones). Off by default, as in Hugging Face transformers; the original Whisper and whisper.cpp turn it on.
    /// </summary>
    public bool ConditionOnPreviousText { get; init; }

    /// <summary>Token selection of the text decoder; greedy by default, as recognizers are evaluated. Ignored with beam search.</summary>
    public SamplingOptions Sampling { get; init; } = SamplingOptions.Greedy;

    /// <summary>
    /// Beams of a beam search over the text tokens (scores normalized by length, as in Hugging Face transformers' <c>num_beams</c>); 1 (the
    /// default) decodes with <see cref="Sampling"/>. More beams cost one decoder pass per beam and token.
    /// </summary>
    public int BeamSize { get; init; } = 1;

    /// <summary>
    /// Long recordings: find the stretches of speech first with <see cref="VoiceActivityDetector"/> and transcribe only those, grouped into windows of at most
    /// 30 seconds that start at speech (no fixed 30 s grid over silence; fewer hallucinated words in pauses, less encoder work). Segment times stay relative to the whole
    /// recording. Needs <see cref="VoiceActivityDetector"/>. Off by default.
    /// </summary>
    public bool UseVoiceActivityDetection { get; init; }

    /// <summary>
    /// The detector for <see cref="UseVoiceActivityDetection"/> (a client of kind <c>voice-activity</c>, e.g. Silero VAD); the caller owns it. When it is <see langword="null"/>
    /// the recognizer resolves an <see cref="IVoiceActivityDetector"/> from the container (see <see cref="VoiceActivityDetectorKey"/>).
    /// </summary>
    public IVoiceActivityDetector? VoiceActivityDetector { get; init; }

    /// <summary>
    /// The service key of the <see cref="IVoiceActivityDetector"/> to resolve from the container (registered with <c>AddTaskClient&lt;IVoiceActivityDetector&gt;(configure, key)</c>),
    /// or <see langword="null"/> for the unkeyed one. Ignored when <see cref="VoiceActivityDetector"/> is set.
    /// </summary>
    public object? VoiceActivityDetectorKey { get; init; }

    /// <summary>How the detector's probabilities become speech segments for <see cref="UseVoiceActivityDetection"/>, or <see langword="null"/> for the defaults.</summary>
    public VoiceActivityOptions? VoiceActivity { get; init; }

    /// <summary>The defaults.</summary>
    public static SpeechRecognitionOptions Default { get; } = new();
}
