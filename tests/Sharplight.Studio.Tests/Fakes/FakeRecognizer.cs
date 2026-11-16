using System.Runtime.CompilerServices;
using Sharplight.Audio;
using Sharplight.Speech;
using Sharplight.Tasks;

namespace Sharplight.Studio.Tests.Fakes;

/// <summary>A speech recognizer that streams scripted segments and records what it was asked.</summary>
internal sealed class FakeRecognizer : ISpeechRecognizer
{
    public TaskClientMetadata Metadata { get; } = new(Sharplight.Models.ModelKinds.SpeechRecognition, "fake-whisper", "whisper");

    public int SampleRate => 16000;

    /// <summary>The segments every request streams.</summary>
    public List<TranscriptionSegment> Script { get; } =
    [
        new() { Start = TimeSpan.Zero, End = TimeSpan.FromSeconds(2.5), Text = " And so, my fellow Americans,", Language = "en" },
        new() { Start = TimeSpan.FromSeconds(2.5), End = TimeSpan.FromSeconds(6), Text = " ask not what your country can do for you.", Language = "en" },
    ];

    /// <summary>When set, the stream waits on it after the first segment (until cancelled).</summary>
    public TaskCompletionSource? Gate { get; set; }

    /// <summary>Blocking work per step (a stand-in for a graph run that holds its thread), or zero.</summary>
    public TimeSpan StepWork { get; set; }

    /// <summary>When set, the stream throws it after the first segment.</summary>
    public Exception? Failure { get; set; }

    /// <summary>The options of every request.</summary>
    public List<SpeechRecognitionOptions?> Requests { get; } = [];

    public Task<Transcription> TranscribeAsync(AudioBuffer audio, SpeechRecognitionOptions? options = null, CancellationToken cancellationToken = default)
        => throw new NotSupportedException("The page streams.");

    public async IAsyncEnumerable<TranscriptionSegment> TranscribeStreamingAsync(AudioBuffer audio, SpeechRecognitionOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        Requests.Add(options);
        for (int i = 0; i < Script.Count; i++)
        {
            await Task.Yield();
            cancellationToken.ThrowIfCancellationRequested();
            SlowWork.Block(StepWork);
            yield return Script[i];
            if (i == 0)
            {
                if (Gate is { } gate)
                {
                    await gate.Task.WaitAsync(cancellationToken);
                }

                if (Failure is { } failure)
                {
                    throw failure;
                }
            }
        }
    }

    public object? GetService(Type serviceType, object? serviceKey = null) => null;

    public void Dispose()
    {
    }
}
