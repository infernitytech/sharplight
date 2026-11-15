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

    /// <summary>Token selection of models that generate codec or acoustic tokens autoregressively.</summary>
    public SamplingOptions Sampling { get; init; } = new();

    /// <summary>The defaults.</summary>
    public static SpeechSynthesisOptions Default { get; } = new();
}
