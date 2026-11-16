using Sharplight.Audio;

namespace Sharplight.Studio.Tests.Fakes;

/// <summary>A capture stream that delivers a fixed number of samples once.</summary>
internal sealed class FakeAudioInput(int sampleRate, int channels, float[] samples) : IAudioInput
{
    private int _position;

    public int SampleRate => sampleRate;

    public int Channels => channels;

    public int AvailableFrames => (samples.Length - _position) / channels;

    public AudioStreamStatistics Statistics => new(_position / channels, 0, 0);

    public bool Started { get; private set; }

    public bool Disposed { get; private set; }

    public void Start() => Started = true;

    public void Stop() => Started = false;

    public int Read(Span<float> destination)
    {
        int count = Math.Min(destination.Length, samples.Length - _position);
        samples.AsSpan(_position, count).CopyTo(destination);
        _position += count;
        return count / channels;
    }

    public void Dispose() => Disposed = true;
}
