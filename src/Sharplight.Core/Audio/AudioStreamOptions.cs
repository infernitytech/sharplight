namespace Sharplight.Audio;

/// <summary>Settings for opening a playback or capture stream.</summary>
public sealed class AudioStreamOptions
{
    /// <summary>Frames per second of the samples exchanged with the stream (the system converts to the device's rate). Default 48000.</summary>
    public int SampleRate { get; set; } = 48000;

    /// <summary>Channels per frame of the samples exchanged with the stream. Default 2.</summary>
    public int Channels { get; set; } = 2;

    /// <summary>Capacity of the buffer between the application and the device. Default 500 ms.</summary>
    public TimeSpan BufferDuration { get; set; } = TimeSpan.FromMilliseconds(500);

    /// <summary>Name of the device to use (<see cref="AudioDeviceInfo.Name"/>), or <see langword="null"/> for the system default.</summary>
    public string? DeviceName { get; set; }
}
