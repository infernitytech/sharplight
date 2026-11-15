using System.Runtime.CompilerServices;
using Sharplight.Audio;
using Sharplight.Speech;
using Sharplight.Tasks;

namespace Sharplight.Serving.Tests;

/// <summary>A recognizer that reports what it received: two segments over the audio, the second naming the request's language and task.</summary>
internal sealed class FakeSpeechRecognizer : ISpeechRecognizer
{
    public TaskClientMetadata Metadata { get; } = new("speech-recognition", "fake-whisper", "fake");

    public int SampleRate => 16000;

    public SpeechRecognitionOptions? LastOptions { get; private set; }

    public AudioBuffer? LastAudio { get; private set; }

    public object? GetService(Type serviceType, object? serviceKey = null) => null;

    public async Task<Transcription> TranscribeAsync(AudioBuffer audio, SpeechRecognitionOptions? options = null, CancellationToken cancellationToken = default)
    {
        var segments = new List<TranscriptionSegment>();
        await foreach (var segment in TranscribeStreamingAsync(audio, options, cancellationToken))
        {
            segments.Add(segment);
        }

        return new Transcription(string.Concat(segments.Select(s => s.Text)).Trim(), options?.Language ?? "en", segments, audio.Duration);
    }

    public async IAsyncEnumerable<TranscriptionSegment> TranscribeStreamingAsync(AudioBuffer audio, SpeechRecognitionOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        LastOptions = options;
        LastAudio = audio;
        await Task.Yield();
        var half = audio.Duration / 2;
        yield return new TranscriptionSegment { Start = TimeSpan.Zero, End = half, Text = " Hello", Tokens = [1, 2] };
        yield return new TranscriptionSegment
        {
            Start = half,
            End = audio.Duration,
            Text = $" {(options?.Translate == true ? "translated" : "world")}.",
            Tokens = [3],
            AverageLogProbability = -0.25,
        };
    }

    public void Dispose()
    {
    }
}
