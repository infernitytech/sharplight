using Sharplight.Audio.Native;

namespace Sharplight.Audio;

/// <summary>Reads stream counters.</summary>
internal static class NativeAudioStatistics
{
    /// <summary>Snapshot of the counters of a stream.</summary>
    /// <param name="stream">The stream handle.</param>
    public static AudioStreamStatistics Read(NativeStreamHandle stream)
    {
        using HandleLease lease = stream.Lease();
        NativeException.ThrowIfFailed(NativeMethods.sla_stream_stats_get(lease.Value, out NativeStreamStats stats), "Reading audio statistics");
        return new AudioStreamStatistics((long)stats.FramesTransferred, (long)stats.Underruns, (long)stats.Overruns);
    }
}
