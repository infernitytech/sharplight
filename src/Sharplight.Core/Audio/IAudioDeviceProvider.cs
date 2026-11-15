namespace Sharplight.Audio;

/// <summary>Access to the machine's audio devices.</summary>
public interface IAudioDeviceProvider
{
    /// <summary>Whether an audio backend was found; false on machines without audio support (servers, containers).</summary>
    public bool IsAvailable { get; }

    /// <summary>Names the backend in use (for example PulseAudio, WASAPI, Core Audio), or says why <see cref="IsAvailable"/> is false.</summary>
    public string Description { get; }

    /// <summary>Lists the devices of one kind; empty when <see cref="IsAvailable"/> is false.</summary>
    /// <param name="kind">Playback or capture.</param>
    public IReadOnlyList<AudioDeviceInfo> GetDevices(AudioDeviceKind kind);

    /// <summary>Opens a playback stream (stopped; call <see cref="IAudioOutput.Start"/>).</summary>
    /// <param name="options">Format, buffer size and device; <see langword="null"/> selects the defaults.</param>
    /// <exception cref="InvalidOperationException">No audio backend, or the device could not be opened.</exception>
    public IAudioOutput OpenOutput(AudioStreamOptions? options = null);

    /// <summary>Opens a capture stream (stopped; call <see cref="IAudioInput.Start"/>).</summary>
    /// <param name="options">Format, buffer size and device; <see langword="null"/> selects the defaults.</param>
    /// <exception cref="InvalidOperationException">No audio backend, or the device could not be opened.</exception>
    public IAudioInput OpenInput(AudioStreamOptions? options = null);
}
