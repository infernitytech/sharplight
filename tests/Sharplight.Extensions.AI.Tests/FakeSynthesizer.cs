using System.Runtime.CompilerServices;
using Sharplight.Audio;
using Sharplight.Models;
using Sharplight.Speech;
using Sharplight.Tasks;

namespace Sharplight.Extensions.AI.Tests;

/// <summary>A synthesizer that produces one 10 ms chunk of a constant level per character.</summary>
internal sealed class FakeSynthesizer : ISpeechSynthesizer
{
    public SpeechSynthesisOptions? LastOptions { get; private set; }

    public TaskClientMetadata Metadata { get; } = new(ModelKinds.SpeechSynthesis, "fake-tts", "fake");

    public int SampleRate => 8000;

    public IReadOnlyList<string> Voices => ["alice", "bob"];

    public object? GetService(Type serviceType, object? serviceKey = null) => null;

    public async Task<SpeechSynthesisResult> SynthesizeAsync(string text, SpeechSynthesisOptions? options = null, CancellationToken cancellationToken = default)
    {
        var samples = new List<float>();
        await foreach (var chunk in SynthesizeStreamingAsync(text, options, cancellationToken))
        {
            samples.AddRange(chunk.Samples);
        }

        return new SpeechSynthesisResult(new AudioBuffer([.. samples], SampleRate), options?.Voice);
    }

    public async IAsyncEnumerable<AudioBuffer> SynthesizeStreamingAsync(string text, SpeechSynthesisOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        LastOptions = options;
        await Task.Yield();
        foreach (char _ in text)
        {
            yield return new AudioBuffer(Enumerable.Repeat(0.5f, SampleRate / 100).ToArray(), SampleRate);
        }
    }

    public void Dispose()
    {
    }
}
