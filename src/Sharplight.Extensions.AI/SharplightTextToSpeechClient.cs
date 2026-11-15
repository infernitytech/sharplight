using System.Runtime.CompilerServices;
using Microsoft.Extensions.AI;
using Sharplight.Audio;
using Sharplight.Speech;

namespace Sharplight.Extensions.AI;

/// <summary>
/// <see cref="ITextToSpeechClient"/> over a Sharplight <see cref="ISpeechSynthesizer"/>. Audio comes back as WAVE (<c>audio/wav</c>,
/// 16-bit PCM) <see cref="DataContent"/>; streaming yields one WAVE chunk per synthesized piece
/// (<see cref="TextToSpeechResponseUpdateKind.AudioUpdating"/>). <see cref="TextToSpeechOptions.VoiceId"/>,
/// <see cref="TextToSpeechOptions.Language"/> and <see cref="TextToSpeechOptions.Speed"/> map to <see cref="SpeechSynthesisOptions"/>.
/// </summary>
public sealed class SharplightTextToSpeechClient : ITextToSpeechClient
{
    private const string WavMediaType = "audio/wav";
    private readonly ISpeechSynthesizer _synthesizer;
    private readonly bool _ownsSynthesizer;
    private readonly TextToSpeechClientMetadata _metadata;

    /// <summary>Wraps <paramref name="synthesizer"/>; disposes it with the client when <paramref name="ownsSynthesizer"/>.</summary>
    /// <param name="synthesizer">The synthesizer.</param>
    /// <param name="ownsSynthesizer">Whether disposing the client disposes the synthesizer.</param>
    public SharplightTextToSpeechClient(ISpeechSynthesizer synthesizer, bool ownsSynthesizer = false)
    {
        ArgumentNullException.ThrowIfNull(synthesizer);
        _synthesizer = synthesizer;
        _ownsSynthesizer = ownsSynthesizer;
        _metadata = new TextToSpeechClientMetadata(TaskClientMetadataNames.Provider, null, synthesizer.Metadata.ModelName);
    }

    /// <inheritdoc/>
    /// <exception cref="NotSupportedException"><see cref="TextToSpeechOptions.AudioFormat"/> asks for a format other than WAVE.</exception>
    public async Task<TextToSpeechResponse> GetAudioAsync(string text, TextToSpeechOptions? options = null, CancellationToken cancellationToken = default)
    {
        var result = await _synthesizer.SynthesizeAsync(text, ToSynthesisOptions(options), cancellationToken).ConfigureAwait(false);
        return new TextToSpeechResponse([ToContent(result.Audio)])
        {
            ModelId = options?.ModelId ?? _metadata.DefaultModelId,
            RawRepresentation = result,
        };
    }

    /// <inheritdoc/>
    /// <exception cref="NotSupportedException"><see cref="TextToSpeechOptions.AudioFormat"/> asks for a format other than WAVE.</exception>
    public async IAsyncEnumerable<TextToSpeechResponseUpdate> GetStreamingAudioAsync(string text, TextToSpeechOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        string? model = options?.ModelId ?? _metadata.DefaultModelId;
        await foreach (var chunk in _synthesizer.SynthesizeStreamingAsync(text, ToSynthesisOptions(options), cancellationToken).ConfigureAwait(false))
        {
            yield return new TextToSpeechResponseUpdate([ToContent(chunk)])
            {
                Kind = TextToSpeechResponseUpdateKind.AudioUpdating,
                ModelId = model,
                RawRepresentation = chunk,
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

        if (serviceType == typeof(TextToSpeechClientMetadata))
        {
            return _metadata;
        }

        return serviceType.IsInstanceOfType(this) ? this : serviceType.IsInstanceOfType(_synthesizer) ? _synthesizer : _synthesizer.GetService(serviceType);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_ownsSynthesizer)
        {
            _synthesizer.Dispose();
        }
    }

    private static DataContent ToContent(AudioBuffer audio)
    {
        using var stream = new MemoryStream();
        WavFile.Write(stream, audio, WavSampleFormat.Pcm16);
        return new DataContent(stream.ToArray(), WavMediaType);
    }

    private static SpeechSynthesisOptions ToSynthesisOptions(TextToSpeechOptions? options)
    {
        if (options is null)
        {
            return SpeechSynthesisOptions.Default;
        }

        if (options.AudioFormat is { } format && format != WavMediaType && format != "wav")
        {
            throw new NotSupportedException($"Audio format '{format}' is not supported; Sharplight returns {WavMediaType}.");
        }

        return new SpeechSynthesisOptions { Voice = options.VoiceId, Language = options.Language, Speed = options.Speed ?? 1f };
    }
}
