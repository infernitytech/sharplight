using Sharplight.Audio;

namespace Sharplight.Studio.Services.Audio;

/// <summary>An entry of the device lists in the settings: the system default (no name) or a named device.</summary>
/// <param name="Name">The device name stored in the settings, or null for the system default.</param>
/// <param name="Title">The text shown in the list.</param>
internal sealed record AudioDeviceChoice(string? Name, string Title)
{
    /// <summary>The system default device.</summary>
    public static AudioDeviceChoice SystemDefault { get; } = new(null, "System default");

    /// <summary>The system default followed by the devices of one kind.</summary>
    /// <param name="devices">The device provider.</param>
    /// <param name="kind">Playback or capture.</param>
    public static IReadOnlyList<AudioDeviceChoice> List(IAudioDeviceProvider devices, AudioDeviceKind kind)
        => [SystemDefault, .. devices.GetDevices(kind).Select(device => new AudioDeviceChoice(device.Name, device.IsDefault ? $"{device.Name} (default)" : device.Name))];

    /// <summary>The configured device when it still exists, else null (the system default): an unplugged device must not break playback.</summary>
    /// <param name="devices">The device provider.</param>
    /// <param name="kind">Playback or capture.</param>
    /// <param name="configured">The device name from the settings.</param>
    public static string? Resolve(IAudioDeviceProvider devices, AudioDeviceKind kind, string? configured)
        => configured is not null && devices.GetDevices(kind).Any(device => device.Name == configured) ? configured : null;
}
