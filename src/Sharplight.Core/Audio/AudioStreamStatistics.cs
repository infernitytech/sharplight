namespace Sharplight.Audio;

/// <summary>Counters of an open audio stream.</summary>
/// <param name="FramesTransferred">Frames moved between the buffer and the device so far.</param>
/// <param name="Underruns">Playback: frames the device needed but the buffer did not have (silence was played).</param>
/// <param name="Overruns">Capture: frames the device delivered but the buffer had no room for (dropped).</param>
public sealed record AudioStreamStatistics(long FramesTransferred, long Underruns, long Overruns);
