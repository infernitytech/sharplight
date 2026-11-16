using Sharplight.Audio;

namespace Sharplight.Studio.Tests.Fakes;

/// <summary>A device provider with a fixed list of devices; streams cannot be opened.</summary>
internal sealed class FakeAudioDevices : IAudioDeviceProvider
{
    public bool IsAvailable { get; set; } = true;

    public string Description => IsAvailable ? "Fake audio" : "No audio support";

    public IReadOnlyList<AudioDeviceInfo> GetDevices(AudioDeviceKind kind) => IsAvailable
        ? [new AudioDeviceInfo(kind, kind == AudioDeviceKind.Playback ? "Speakers" : "Microphone", true), new AudioDeviceInfo(kind, kind == AudioDeviceKind.Playback ? "Headphones" : "Headset", false)]
        : [];

    /// <summary>The options of the last stream that was opened.</summary>
    public AudioStreamOptions? LastOptions { get; private set; }

    /// <summary>The last playback stream.</summary>
    public FakeAudioOutput? Output { get; private set; }

    /// <summary>What a capture stream delivers.</summary>
    public float[] Microphone { get; set; } = [];

    public IAudioOutput OpenOutput(AudioStreamOptions? options = null)
    {
        LastOptions = options;
        Output = new FakeAudioOutput(options?.SampleRate ?? 48000, options?.Channels ?? 2);
        return Output;
    }

    public IAudioInput OpenInput(AudioStreamOptions? options = null)
    {
        LastOptions = options;
        return new FakeAudioInput(options?.SampleRate ?? 48000, options?.Channels ?? 1, Microphone);
    }
}
