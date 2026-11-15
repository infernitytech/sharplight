using System.Diagnostics;

namespace Sharplight.Studio.Services.Audio;

/// <summary>
/// Records the microphone with the recorder the system has: on Linux <c>parecord</c> (PulseAudio and PipeWire), <c>pw-record</c> or
/// <c>arecord</c>, on macOS <c>ffmpeg</c> (AVFoundation); Windows needs <c>ffmpeg</c> with the default DirectShow device named in
/// <c>SHARPLIGHT_STUDIO_MIC</c>. Without one the page takes audio files only.
/// </summary>
internal sealed class ProcessAudioRecorder : IAudioRecorder
{
    private const int SampleRate = 16000;
    private readonly string? _program;
    private readonly Func<string, IEnumerable<string>>? _arguments;

    public ProcessAudioRecorder()
    {
        (_program, _arguments) = Detect();
    }

    /// <inheritdoc/>
    public bool IsAvailable => _program is not null;

    /// <inheritdoc/>
    public string Description => _program is null
        ? "No recorder found (install pulseaudio-utils, pipewire, alsa-utils or ffmpeg); choose an audio file instead."
        : $"Records with {Path.GetFileNameWithoutExtension(_program)}.";

    /// <inheritdoc/>
    public IRecording Start(string path)
    {
        if (_program is null || _arguments is null)
        {
            throw new NotSupportedException(Description);
        }

        return new Recording(SystemAudioTools.Start(_program, _arguments(path)));
    }

    private static (string? Program, Func<string, IEnumerable<string>>? Arguments) Detect()
    {
        string rate = SampleRate.ToString(System.Globalization.CultureInfo.InvariantCulture);
        if (OperatingSystem.IsMacOS())
        {
            return Pick(("ffmpeg", path => ["-y", "-loglevel", "error", "-f", "avfoundation", "-i", ":0", "-ac", "1", "-ar", rate, path]));
        }

        if (OperatingSystem.IsWindows())
        {
            string? device = Environment.GetEnvironmentVariable("SHARPLIGHT_STUDIO_MIC");
            return device is null
                ? (null, null)
                : Pick(("ffmpeg", path => ["-y", "-loglevel", "error", "-f", "dshow", "-i", $"audio={device}", "-ac", "1", "-ar", rate, path]));
        }

        return Pick(
            ("parecord", path => ["--file-format=wav", "--format=s16le", $"--rate={rate}", "--channels=1", path]),
            ("pw-record", path => ["--rate", rate, "--channels", "1", "--format", "s16", path]),
            ("arecord", path => ["-q", "-f", "S16_LE", "-r", rate, "-c", "1", "-t", "wav", path]));
    }

    private static (string? Program, Func<string, IEnumerable<string>>? Arguments) Pick(params (string Program, Func<string, IEnumerable<string>> Arguments)[] candidates)
    {
        foreach (var (program, arguments) in candidates)
        {
            if (SystemAudioTools.Find(program) is { } found)
            {
                return (found, arguments);
            }
        }

        return (null, null);
    }

    private sealed class Recording(Process process) : IRecording
    {
        /// <inheritdoc/>
        public async Task StopAsync()
        {
            if (process.HasExited)
            {
                return;
            }

            // SIGINT lets the recorder finalize the WAV header; Kill would leave the file with a zero length.
            if (OperatingSystem.IsWindows())
            {
                process.Kill();
            }
            else
            {
                Interrupt(process.Id);
            }

            using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(3));
            try
            {
                await process.WaitForExitAsync(timeout.Token).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                process.Kill();
            }
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            if (!process.HasExited)
            {
                process.Kill();
            }

            process.Dispose();
        }

        private static void Interrupt(int id)
        {
            using var kill = Process.Start(new ProcessStartInfo("kill") { ArgumentList = { "-INT", id.ToString(System.Globalization.CultureInfo.InvariantCulture) }, UseShellExecute = false });
            kill?.WaitForExit();
        }
    }
}
