using System.Diagnostics;
using Microsoft.Extensions.DependencyInjection;
using Sharplight.Audio;
using Sharplight.Runtime;
using Sharplight.Samples;
using Sharplight.Speech;
using Sharplight.Tasks;

namespace Sharplight.Transcribe;

/// <summary>
/// Runs <c>sharplight-transcribe</c>: registers Sharplight and one <see cref="ISpeechRecognizer"/> for the model with the service container
/// (the model loads on first use), reads the WAV file, transcribes it and writes the transcript.
/// </summary>
public static class TranscribeApplication
{
    /// <summary>Transcribes the file the settings name.</summary>
    /// <param name="settings">The parsed command line.</param>
    /// <param name="cancellationToken">Stops recognition between decoding steps.</param>
    /// <returns>The process exit code.</returns>
    public static async Task<int> RunAsync(TranscribeSettings settings, CancellationToken cancellationToken)
    {
        if (!File.Exists(settings.AudioPath))
        {
            throw new SampleException($"Audio file '{settings.AudioPath}' not found.", ExitCodes.InvalidInput);
        }

        var audio = WavFile.Read(settings.AudioPath);
        var services = new ServiceCollection().AddSharplight();
        var provider = SampleRuntime.SelectProvider(settings.UseCpu);
        using var device = provider.OpenDevice(SampleRuntime.GetDevice(provider, settings.DeviceIndex));
        services.AddSpeechRecognizer(options =>
        {
            options.ModelPath = settings.ModelPath;
            options.Devices = [new DeviceSlot { Device = device }];
            options.Log = settings.Verbose ? message => Console.Error.WriteLine(message) : null;
        });
        await using var container = services.BuildServiceProvider();
        var recognizer = container.GetRequiredService<ISpeechRecognizer>();

        var stopwatch = Stopwatch.StartNew();
        var transcription = await recognizer.TranscribeAsync(audio, settings.Recognition, cancellationToken);
        if (settings.Verbose)
        {
            await Console.Error.WriteLineAsync($"Transcribed {audio.Duration.TotalSeconds:F1} s of audio in {stopwatch.Elapsed.TotalSeconds:F2} s "
                + $"({recognizer.Metadata.ModelName}, language {transcription.Language ?? "unknown"}).");
        }

        await TranscriptWriter.WriteAsync(transcription, settings.Format, Console.Out);
        return ExitCodes.Success;
    }
}
