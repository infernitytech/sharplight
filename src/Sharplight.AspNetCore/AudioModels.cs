using Sharplight.Speech;

namespace Sharplight.AspNetCore;

/// <summary>The speech-recognition models a host serves on the audio endpoints, by the name clients pass as <c>model</c>.</summary>
public sealed class AudioModels
{
    /// <summary>Speech recognizers by name.</summary>
    public Dictionary<string, ISpeechRecognizer> Recognizers { get; } = new(StringComparer.Ordinal);

    /// <summary>Recognizer used when a request names none or an unknown one while only one is registered (default: the first registered).</summary>
    public string? DefaultRecognizer { get; set; }

    /// <summary>Speech synthesizers by name (<c>/v1/audio/speech</c>).</summary>
    public Dictionary<string, ISpeechSynthesizer> Synthesizers { get; } = new(StringComparer.Ordinal);

    /// <summary>Synthesizer used when a request names none or an unknown one while only one is registered (default: the first registered).</summary>
    public string? DefaultSynthesizer { get; set; }

    /// <summary>Largest accepted request body (the uploaded audio).</summary>
    public long MaxRequestBodyBytes { get; set; } = 64 * 1024 * 1024;
}
