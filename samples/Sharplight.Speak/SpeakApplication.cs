using System.Diagnostics;
using Microsoft.Extensions.DependencyInjection;
using Sharplight.Audio;
using Sharplight.Llm;
using Sharplight.Runtime;
using Sharplight.Samples;
using Sharplight.Speech;
using Sharplight.Vision;

namespace Sharplight.Speak;

/// <summary>Loads a speech-synthesis model through the task layer, speaks the text and writes the WAV file.</summary>
internal static class SpeakApplication
{
    public static async Task<int> RunAsync(SpeakSettings settings, CancellationToken cancellationToken)
    {
        var progress = new ConsoleLoadProgress();
        using var runtime = new SampleRuntime(services => services.AddSpeechSynthesizer((options, container) =>
        {
            var provider = SampleRuntime.SelectProvider(container.GetRequiredService<ExecutionProviderCatalog>(), settings.UseCpu);
            var device = provider.OpenDevice(SampleRuntime.GetDevice(provider, settings.DeviceIndex));
            options.ModelPath = settings.ModelPath;
            options.Devices = [new DeviceSlot { Device = device }];
            if (settings.VocoderPath is { } vocoder)
            {
                options.Auxiliary[AuxiliaryRoles.Vocoder] = vocoder;
            }

            options.Log = progress.Log;
            options.Progress = progress;
        }));
        var container = runtime.Services;

        var stopwatch = Stopwatch.StartNew();
        var synthesizer = container.GetRequiredService<ISpeechSynthesizer>();
        Console.Error.WriteLine($"Loaded {synthesizer.Metadata.ModelName} ({synthesizer.SampleRate} Hz) in {stopwatch.Elapsed.TotalSeconds:F1} s; voices: {(synthesizer.Voices.Count == 0 ? "-" : string.Join(", ", synthesizer.Voices))}");

        var options = new SpeechSynthesisOptions
        {
            Voice = settings.Voice,
            Language = settings.Language,
            Speed = settings.Speed,
            ReferenceAudio = settings.ReferencePath is { } reference ? WavFile.Read(reference) : null,
            ReferenceText = settings.ReferenceText,
            // Without --greedy only the seed is set: each model then samples with its own settings.
            Sampling = settings.Greedy
                ? new SamplingOptions { Temperature = 0f, TopK = 1, TopP = 0.9f, RepetitionPenalty = 1f, Seed = settings.Seed }
                : new SamplingOptions { Seed = settings.Seed },
        };
        stopwatch.Restart();
        AudioBuffer audio;
        if (settings.Stream)
        {
            var segments = new List<float>();
            TimeSpan? first = null;
            await foreach (var segment in synthesizer.SynthesizeStreamingAsync(settings.Text, options, cancellationToken))
            {
                first ??= stopwatch.Elapsed;
                segments.AddRange(segment.Samples);
            }

            Console.Error.WriteLine($"First segment after {first?.TotalSeconds ?? 0:F2} s");
            audio = new AudioBuffer([.. segments], synthesizer.SampleRate);
        }
        else
        {
            audio = (await synthesizer.SynthesizeAsync(settings.Text, options, cancellationToken)).Audio;
        }

        double seconds = stopwatch.Elapsed.TotalSeconds;
        WavFile.Write(settings.OutputPath, audio);
        Console.WriteLine($"Wrote {settings.OutputPath}: {audio.Duration.TotalSeconds:F2} s of audio in {seconds:F2} s (real-time factor {seconds / Math.Max(1e-9, audio.Duration.TotalSeconds):F2}).");
        return ExitCodes.Success;
    }
}
