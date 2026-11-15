using System.Runtime.CompilerServices;
using System.Text;
using Sharplight.Audio;
using Sharplight.Runtime;
using Sharplight.Speech;
using Sharplight.Tasks;

namespace Sharplight.Package.Tests;

/// <summary>
/// A speech recognizer as an architecture package would register one, reduced to what the task layer must provide: it compiles the
/// model's encoder component, runs it on the first samples of the audio, and reports the tokenizer attachment and the encoder output
/// sum as its text.
/// </summary>
public sealed class EncoderOnlyRecognizer(TaskModel model) : ISpeechRecognizer
{
    public const int Frames = 2;
    public const int FeatureSize = 8;

    public TaskClientMetadata Metadata { get; } = TaskClientMetadata.For(model.Definition);

    public int SampleRate => model.Description.ConfigValue("sampleRate", 16000);

    public bool Disposed { get; private set; }

    public object? GetService(Type serviceType, object? serviceKey = null) => serviceType.IsInstanceOfType(model) ? model : null;

    public Task<Transcription> TranscribeAsync(AudioBuffer audio, SpeechRecognitionOptions? options = null, CancellationToken cancellationToken = default)
    {
        using var encoder = model.CompileComponent("encoder");
        var run = new RunParameters();
        run.Symbols["F"] = Frames;
        encoder.SetInput<float>(encoder.Graph.Inputs[0], audio.Samples.AsSpan(0, Frames * FeatureSize));
        encoder.Run(run);
        var hidden = new float[Frames * 4];
        encoder.GetOutput("hidden", System.Runtime.InteropServices.MemoryMarshal.AsBytes(hidden.AsSpan()));
        string vocab = Encoding.UTF8.GetString(model.ReadAttachment("tokenizer", "vocab"));
        string text = string.Create(System.Globalization.CultureInfo.InvariantCulture, $"{vocab}:{hidden.Sum():F3}");
        var segment = new TranscriptionSegment { Start = TimeSpan.Zero, End = audio.Duration, Text = text };
        return Task.FromResult(new Transcription(text, null, [segment], audio.Duration));
    }

    public async IAsyncEnumerable<TranscriptionSegment> TranscribeStreamingAsync(AudioBuffer audio, SpeechRecognitionOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var transcription = await TranscribeAsync(audio, options, cancellationToken);
        foreach (var segment in transcription.Segments)
        {
            yield return segment;
        }
    }

    public void Dispose() => Disposed = true;
}
