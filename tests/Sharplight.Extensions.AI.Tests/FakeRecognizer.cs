using System.Runtime.CompilerServices;
using Sharplight.Audio;
using Sharplight.Models;
using Sharplight.Speech;
using Sharplight.Tasks;

namespace Sharplight.Extensions.AI.Tests;

/// <summary>A recognizer that reports what it was asked: one segment per 100 ms of audio.</summary>
internal sealed class FakeRecognizer : ISpeechRecognizer
{
    public SpeechRecognitionOptions? LastOptions { get; private set; }

    public AudioBuffer? LastAudio { get; private set; }

    public bool Disposed { get; private set; }

    public TaskClientMetadata Metadata { get; } = new(ModelKinds.SpeechRecognition, "fake-whisper", "fake");

    public int SampleRate => 16000;

    public object? GetService(Type serviceType, object? serviceKey = null) => null;

    public async Task<Transcription> TranscribeAsync(AudioBuffer audio, SpeechRecognitionOptions? options = null, CancellationToken cancellationToken = default)
    {
        var segments = new List<TranscriptionSegment>();
        await foreach (var segment in TranscribeStreamingAsync(audio, options, cancellationToken))
        {
            segments.Add(segment);
        }

        return new Transcription(string.Concat(segments.Select(s => s.Text)), options?.Language, segments, audio.Duration);
    }

    public async IAsyncEnumerable<TranscriptionSegment> TranscribeStreamingAsync(AudioBuffer audio, SpeechRecognitionOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        LastAudio = audio;
        LastOptions = options;
        await Task.Yield();
        var step = TimeSpan.FromMilliseconds(100);
        for (var start = TimeSpan.Zero; start < audio.Duration; start += step)
        {
            cancellationToken.ThrowIfCancellationRequested();
            yield return new TranscriptionSegment { Start = start, End = start + step, Text = $"[{start.TotalMilliseconds}]" };
        }
    }

    public void Dispose() => Disposed = true;
}
