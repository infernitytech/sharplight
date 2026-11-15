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

    /// <summary>The defaults.</summary>
    public static SpeechRecognitionOptions Default { get; } = new();
}
