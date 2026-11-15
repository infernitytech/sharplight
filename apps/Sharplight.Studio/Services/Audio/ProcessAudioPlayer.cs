using System.Diagnostics;

namespace Sharplight.Studio.Services.Audio;

/// <summary>
/// Plays WAV files with the player the system has: <c>afplay</c> on macOS, PowerShell's <c>SoundPlayer</c> on Windows, and on Linux
/// <c>pw-play</c>, <c>paplay</c>, <c>aplay</c> or <c>ffplay</c> (the first one installed).
/// </summary>
internal sealed class ProcessAudioPlayer : IAudioPlayer
{
    private readonly Func<string, IEnumerable<string>>? _arguments;
    private readonly string? _program;

    public ProcessAudioPlayer()
    {
        (_program, _arguments) = Detect();
    }

    /// <inheritdoc/>
    public bool IsAvailable => _program is not null;

    /// <inheritdoc/>
    public string Description => _program is null
        ? "No audio player found (install pipewire, pulseaudio, alsa-utils or ffmpeg); the audio is saved and can be opened with the system player."
        : $"Plays with {Path.GetFileNameWithoutExtension(_program)}.";

    /// <inheritdoc/>
    public async Task PlayAsync(string path, CancellationToken cancellationToken)
    {
        if (_program is null || _arguments is null)
        {
            throw new NotSupportedException(Description);
        }

        using var process = SystemAudioTools.Start(_program, _arguments(path));
        try
        {
            await process.WaitForExitAsync(cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            Stop(process);
            throw;
        }
    }

    private static void Stop(Process process)
    {
        try
        {
            process.Kill(entireProcessTree: true);
        }
        catch (InvalidOperationException)
        {
            // Already finished.
        }
    }

    private static (string? Program, Func<string, IEnumerable<string>>? Arguments) Detect()
    {
        if (OperatingSystem.IsMacOS())
        {
            return Pick(("afplay", path => [path]));
        }

        if (OperatingSystem.IsWindows())
        {
            return Pick(("powershell", path => ["-NoProfile", "-Command", $"(New-Object Media.SoundPlayer '{path.Replace("'", "''")}').PlaySync()"]));
        }

        return Pick(("pw-play", path => [path]), ("paplay", path => [path]), ("aplay", path => ["-q", path]),
            ("ffplay", path => ["-nodisp", "-autoexit", "-loglevel", "quiet", path]));
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
}
