using Sharplight.Studio.Services.Audio;

namespace Sharplight.Studio.Tests.Fakes;

/// <summary>Audio output and microphone stand-ins: playback and recording are recorded, nothing is heard.</summary>
internal sealed class FakeAudio : IAudioPlayer, IAudioRecorder
{
    public bool IsAvailable { get; set; } = true;

    public string Description => IsAvailable ? "Fake audio." : "No audio device.";

    /// <summary>Files played so far.</summary>
    public List<string> Played { get; } = [];

    /// <summary>Gates playback: when set, <see cref="PlayAsync"/> waits for it (or cancellation).</summary>
    public TaskCompletionSource? PlayGate { get; set; }

    /// <summary>What a recording writes to the file when it stops.</summary>
    public Action<string>? OnRecorded { get; set; }

    public async Task PlayAsync(string path, CancellationToken cancellationToken)
    {
        Played.Add(path);
        if (PlayGate is { } gate)
        {
            await gate.Task.WaitAsync(cancellationToken);
        }
    }

    public IRecording Start(string path) => new FakeRecording(() => OnRecorded?.Invoke(path));

    private sealed class FakeRecording(Action stopped) : IRecording
    {
        public Task StopAsync()
        {
            stopped();
            return Task.CompletedTask;
        }

        public void Dispose()
        {
        }
    }
}
