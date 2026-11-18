using Sharplight.Speech;

namespace Sharplight.AspNetCore;

/// <summary>
/// The audio-model registrations of <c>services.AddSharplightSpeechRecognizer(...)</c>: recognizers given directly and recognizers resolved from the container on first use; resolved
/// into the singleton <see cref="AudioModels"/> on the first request.
/// </summary>
internal sealed class AudioModelRegistrations
{
    /// <summary>Recognizers given directly, by name.</summary>
    public Dictionary<string, ISpeechRecognizer> Recognizers { get; } = new(StringComparer.Ordinal);

    /// <summary>Recognizers resolved from the container, by name.</summary>
    public Dictionary<string, Func<IServiceProvider, ISpeechRecognizer>> Lazy { get; } = new(StringComparer.Ordinal);

    /// <summary>Synthesizers given directly, by name.</summary>
    public Dictionary<string, ISpeechSynthesizer> Synthesizers { get; } = new(StringComparer.Ordinal);

    /// <summary>The default synthesizer name.</summary>
    public string? DefaultSynthesizer { get; set; }

    /// <summary>The default model name.</summary>
    public string? DefaultRecognizer { get; set; }

    /// <summary>Resolves every registration.</summary>
    /// <param name="provider">The container.</param>
    public AudioModels Resolve(IServiceProvider provider)
    {
        var models = new AudioModels { DefaultRecognizer = DefaultRecognizer, DefaultSynthesizer = DefaultSynthesizer };
        foreach (var (name, synthesizer) in Synthesizers)
        {
            models.Synthesizers[name] = synthesizer;
        }

        foreach (var (name, recognizer) in Recognizers)
        {
            models.Recognizers[name] = recognizer;
        }

        foreach (var (name, resolve) in Lazy)
        {
            models.Recognizers[name] = resolve(provider);
        }

        return models;
    }
}
