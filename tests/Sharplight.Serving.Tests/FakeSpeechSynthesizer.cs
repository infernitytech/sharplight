using System.Runtime.CompilerServices;
using Sharplight.Audio;
using Sharplight.Speech;
using Sharplight.Tasks;

namespace Sharplight.Serving.Tests;

/// <summary>A synthesizer that speaks a constant tone, one chunk of 100 samples per word, and remembers the request.</summary>
internal sealed class FakeSpeechSynthesizer : ISpeechSynthesizer
{
    public TaskClientMetadata Metadata { get; } = new("speech-synthesis", "fake-tts", "fake");

    public int SampleRate => 8000;

    public IReadOnlyList<string> Voices { get; } = ["alloy", "echo"];

    public string? LastText { get; private set; }

    public SpeechSynthesisOptions? LastOptions { get; private set; }

    public object? GetService(Type serviceType, object? serviceKey = null) => null;

    public Task<SpeechSynthesisResult> SynthesizeAsync(string text, SpeechSynthesisOptions? options = null, CancellationToken cancellationToken = default)
    {
        LastText = text;
        LastOptions = options;
        int words = text.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length;
        return Task.FromResult(new SpeechSynthesisResult(new AudioBuffer(Enumerable.Repeat(0.5f, 100 * words).ToArray(), SampleRate), options?.Voice));
    }

    public async IAsyncEnumerable<AudioBuffer> SynthesizeStreamingAsync(string text, SpeechSynthesisOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        LastText = text;
        LastOptions = options;
        foreach (var _ in text.Split(' ', StringSplitOptions.RemoveEmptyEntries))
        {
            await Task.Yield();
            yield return new AudioBuffer(Enumerable.Repeat(0.5f, 100).ToArray(), SampleRate);
        }
    }

    public void Dispose()
    {
    }
}
