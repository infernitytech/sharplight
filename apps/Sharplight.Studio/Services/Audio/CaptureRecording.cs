using Sharplight.Audio;

namespace Sharplight.Studio.Services.Audio;

/// <summary>A running capture: a background loop drains the input buffer; stopping writes the samples as a 16-bit WAV file.</summary>
internal sealed class CaptureRecording : IRecording
{
    private static readonly TimeSpan s_pollInterval = TimeSpan.FromMilliseconds(20);

    private readonly IAudioInput _input;
    private readonly IAudioEncoder _encoder;
    private readonly string _path;
    private readonly List<float> _samples = [];
    private readonly CancellationTokenSource _stop = new();
    private readonly Task _loop;
    private int _finished;

    public CaptureRecording(IAudioInput input, IAudioEncoder encoder, string path)
    {
        _input = input;
        _encoder = encoder;
        _path = path;
        _loop = Task.Run(DrainAsync);
    }

    /// <inheritdoc/>
    public async Task StopAsync()
    {
        if (Interlocked.Exchange(ref _finished, 1) != 0)
        {
            return;
        }

        await _stop.CancelAsync().ConfigureAwait(false);
        await _loop.ConfigureAwait(false);
        _input.Stop();
        Drain();
        _input.Dispose();
        var audio = new AudioBuffer([.. _samples], _input.SampleRate, _input.Channels);
        _encoder.Save(audio, _path);
    }

    /// <inheritdoc/>
    public void Dispose() => StopAsync().GetAwaiter().GetResult();

    private async Task DrainAsync()
    {
        using var timer = new PeriodicTimer(s_pollInterval);
        try
        {
            while (await timer.WaitForNextTickAsync(_stop.Token).ConfigureAwait(false))
            {
                Drain();
            }
        }
        catch (OperationCanceledException)
        {
            // Stopped.
        }
    }

    private void Drain()
    {
        var chunk = new float[4096 * _input.Channels];
        int frames;
        while ((frames = _input.Read(chunk)) > 0)
        {
            _samples.AddRange(chunk.AsSpan(0, frames * _input.Channels).ToArray());
        }
    }
}
