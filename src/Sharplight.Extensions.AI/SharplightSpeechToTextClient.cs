using System.Runtime.CompilerServices;
using Microsoft.Extensions.AI;
using Sharplight.Audio;
using Sharplight.Speech;

namespace Sharplight.Extensions.AI;

/// <summary>
/// <see cref="ISpeechToTextClient"/> over a Sharplight <see cref="ISpeechRecognizer"/>. The audio stream is a WAVE file (PCM or float,
/// any rate and channel count; <see cref="WavFile"/>). <see cref="SpeechToTextOptions.SpeechLanguage"/> sets the spoken language; a
/// <see cref="SpeechToTextOptions.TextLanguage"/> of <c>en</c> for non-English speech asks for translation. Each recognized segment
/// is one streaming update (<see cref="SpeechToTextResponseUpdateKind.TextUpdated"/>) with its start and end time.
/// </summary>
public sealed class SharplightSpeechToTextClient : ISpeechToTextClient
{
    private const string English = "en";
    private readonly ISpeechRecognizer _recognizer;
    private readonly bool _ownsRecognizer;
    private readonly SpeechToTextClientMetadata _metadata;

    /// <summary>Wraps <paramref name="recognizer"/>; disposes it with the client when <paramref name="ownsRecognizer"/>.</summary>
    /// <param name="recognizer">The recognizer.</param>
    /// <param name="ownsRecognizer">Whether disposing the client disposes the recognizer.</param>
    public SharplightSpeechToTextClient(ISpeechRecognizer recognizer, bool ownsRecognizer = false)
    {
        ArgumentNullException.ThrowIfNull(recognizer);
        _recognizer = recognizer;
        _ownsRecognizer = ownsRecognizer;
        _metadata = new SpeechToTextClientMetadata(TaskClientMetadataNames.Provider, null, recognizer.Metadata.ModelName);
    }

    /// <inheritdoc/>
    public async Task<SpeechToTextResponse> GetTextAsync(Stream audioSpeechStream, SpeechToTextOptions? options = null, CancellationToken cancellationToken = default)
    {
        var audio = ReadAudio(audioSpeechStream);
        var transcription = await _recognizer.TranscribeAsync(audio, ToRecognitionOptions(options), cancellationToken).ConfigureAwait(false);
        return new SpeechToTextResponse(transcription.Text)
        {
            StartTime = TimeSpan.Zero,
            EndTime = transcription.Duration,
            ModelId = options?.ModelId ?? _metadata.DefaultModelId,
            RawRepresentation = transcription,
        };
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<SpeechToTextResponseUpdate> GetStreamingTextAsync(Stream audioSpeechStream, SpeechToTextOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var audio = ReadAudio(audioSpeechStream);
        string? model = options?.ModelId ?? _metadata.DefaultModelId;
        await foreach (var segment in _recognizer.TranscribeStreamingAsync(audio, ToRecognitionOptions(options), cancellationToken).ConfigureAwait(false))
        {
            yield return new SpeechToTextResponseUpdate(segment.Text)
            {
                Kind = SpeechToTextResponseUpdateKind.TextUpdated,
                StartTime = segment.Start,
                EndTime = segment.End,
                ModelId = model,
                RawRepresentation = segment,
                AdditionalProperties = segment.Speaker is null ? null : new AdditionalPropertiesDictionary { ["speaker"] = segment.Speaker },
            };
        }
    }

    /// <inheritdoc/>
    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);
        if (serviceKey is not null)
        {
            return null;
        }

        if (serviceType == typeof(SpeechToTextClientMetadata))
        {
            return _metadata;
        }

        return serviceType.IsInstanceOfType(this) ? this : serviceType.IsInstanceOfType(_recognizer) ? _recognizer : _recognizer.GetService(serviceType);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_ownsRecognizer)
        {
            _recognizer.Dispose();
        }
    }

    private static AudioBuffer ReadAudio(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);
        return WavFile.Read(stream);
    }

    private static SpeechRecognitionOptions ToRecognitionOptions(SpeechToTextOptions? options)
    {
        if (options is null)
        {
            return SpeechRecognitionOptions.Default;
        }

        bool translate = options.TextLanguage == English && options.SpeechLanguage is not null && options.SpeechLanguage != English;
        return new SpeechRecognitionOptions { Language = options.SpeechLanguage, Translate = translate };
    }
}
