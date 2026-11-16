using Sharplight.Audio;

namespace Sharplight.Studio.Tests.Fakes;

/// <summary>A playback stream that swallows samples and remembers them.</summary>
internal sealed class FakeAudioOutput(int sampleRate, int channels) : IAudioOutput
{
    public int SampleRate => sampleRate;

    public int Channels => channels;

    public int FreeFrames => int.MaxValue;

    public int QueuedFrames => 0;

    public AudioStreamStatistics Statistics => new(Written.Count / channels, 0, 0);

    public List<float> Written { get; } = [];

    public bool Started { get; private set; }

    public bool Disposed { get; private set; }

    public void Start() => Started = true;

    public void Stop() => Started = false;

    public int Write(ReadOnlySpan<float> samples)
    {
        Written.AddRange(samples.ToArray());
        return samples.Length / channels;
    }

    public void Dispose() => Disposed = true;
}
