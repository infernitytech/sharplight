using System.Diagnostics;
using Microsoft.Extensions.DependencyInjection;
using Sharplight.Audio;
using Sharplight.Llm;
using Sharplight.Runtime;
using Sharplight.Samples;
using Sharplight.Speech;
using Sharplight.Tasks;

namespace Sharplight.Speak;

/// <summary>Loads a speech-synthesis model through the task layer, speaks the text and writes the WAV file.</summary>
internal static class SpeakApplication
{
    public static async Task<int> RunAsync(SpeakSettings settings, CancellationToken cancellationToken)
    {
        var services = new ServiceCollection().AddSharplight();
        var provider = SampleRuntime.SelectProvider(settings.UseCpu);
        var device = provider.OpenDevice(SampleRuntime.GetDevice(provider, settings.DeviceIndex));
        var progress = new ConsoleLoadProgress();
        services.AddSpeechSynthesizer(options =>
        {
            options.ModelPath = settings.ModelPath;
            options.Devices = [new DeviceSlot { Device = device }];
            options.Auxiliary = settings.VocoderPath is { } vocoder ? [("vocoder", vocoder)] : [];
            options.Log = progress.Log;
            options.Progress = progress;
        });
        await using var container = services.BuildServiceProvider();

        var stopwatch = Stopwatch.StartNew();
        var synthesizer = container.GetRequiredService<ISpeechSynthesizer>();
        Console.Error.WriteLine($"Loaded {synthesizer.Metadata.ModelName} ({synthesizer.SampleRate} Hz) in {stopwatch.Elapsed.TotalSeconds:F1} s; voices: {(synthesizer.Voices.Count == 0 ? "-" : string.Join(", ", synthesizer.Voices))}");

        var options = new SpeechSynthesisOptions
        {
            Voice = settings.Voice,
            Sampling = new SamplingOptions
            {
                Temperature = settings.Greedy ? 0f : 0.4f,
                TopK = settings.Greedy ? 1 : 40,
                TopP = 0.9f,
                RepetitionPenalty = settings.Greedy ? 1f : 1.1f,
                Seed = settings.Seed,
            },
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
