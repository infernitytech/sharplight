namespace Sharplight.Audio;

/// <summary>An audio device the system offers.</summary>
/// <param name="Kind">Playback or capture.</param>
/// <param name="Name">The device name as shown to the user; stable enough to store as a preference.</param>
/// <param name="IsDefault">Whether the system uses this device when none is chosen.</param>
public sealed record AudioDeviceInfo(AudioDeviceKind Kind, string Name, bool IsDefault);
