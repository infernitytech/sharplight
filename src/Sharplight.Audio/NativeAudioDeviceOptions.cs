namespace Sharplight.Audio;

/// <summary>Settings of <see cref="NativeAudioDeviceProvider"/>; bindable from configuration.</summary>
public sealed class NativeAudioDeviceOptions
{
    /// <summary>
    /// Uses the null backend: one virtual playback and one virtual capture device that consume and produce silence in real time, without
    /// touching hardware. For tests and headless machines. Default false.
    /// </summary>
    public bool UseNullBackend { get; set; }
}
