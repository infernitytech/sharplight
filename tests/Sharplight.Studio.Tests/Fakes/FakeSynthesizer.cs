using System.Runtime.CompilerServices;
using Sharplight.Audio;
using Sharplight.Speech;
using Sharplight.Tasks;

namespace Sharplight.Studio.Tests.Fakes;

/// <summary>A speech synthesizer that returns a sine tone and records what it was asked.</summary>
internal sealed class FakeSynthesizer : ISpeechSynthesizer
{
    public TaskClientMetadata Metadata { get; } = new(Sharplight.Models.ModelKinds.SpeechSynthesis, "fake-tts", "outetts");

    public int SampleRate => 24000;

    public IReadOnlyList<string> Voices { get; set; } = [];

    /// <summary>Blocking work per step (a stand-in for a graph run that holds its thread), or zero.</summary>
    public TimeSpan StepWork { get; set; }

    /// <summary>Texts and options of every request.</summary>
    public List<(string Text, SpeechSynthesisOptions? Options, bool Streaming)> Requests { get; } = [];

    /// <summary>A tone of <paramref name="seconds"/>.</summary>
    public AudioBuffer Tone(double seconds) => new([.. Enumerable.Range(0, (int)(SampleRate * seconds)).Select(i => 0.3f * MathF.Sin(2 * MathF.PI * 440 * i / SampleRate))], SampleRate);

    public Task<SpeechSynthesisResult> SynthesizeAsync(string text, SpeechSynthesisOptions? options = null, CancellationToken cancellationToken = default)
    {
        Requests.Add((text, options, false));
        return Task.FromResult(new SpeechSynthesisResult(Tone(1), options?.Voice));
    }

    public async IAsyncEnumerable<AudioBuffer> SynthesizeStreamingAsync(string text, SpeechSynthesisOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        Requests.Add((text, options, true));
        foreach (var word in text.Split(' ', StringSplitOptions.RemoveEmptyEntries))
        {
            await Task.Yield();
            cancellationToken.ThrowIfCancellationRequested();
            SlowWork.Block(StepWork);
            yield return Tone(0.2);
        }
    }

    public object? GetService(Type serviceType, object? serviceKey = null) => null;

    public void Dispose()
    {
    }
}
