using Sharplight.Shaders.Lowering;

namespace Sharplight.Vulkan;

/// <summary>
/// Decides where a command buffer needs memory barriers: a launch needs one before it when it reads data written, or
/// writes data read or written, by a launch since the previous barrier.
/// </summary>
internal sealed class HazardTracker
{
    private readonly List<BufferRange> _pendingWrites = [];
    private readonly List<BufferRange> _pendingReads = [];

    /// <summary>True when <paramref name="launch"/> conflicts with a launch recorded since the last barrier.</summary>
    public bool Conflicts(KernelLaunch launch)
        => launch.Reads.Any(r => _pendingWrites.Any(w => w.Overlaps(r)))
           || launch.Writes.Any(w => _pendingWrites.Any(p => p.Overlaps(w)) || _pendingReads.Any(p => p.Overlaps(w)));

    /// <summary>Forgets all pending accesses (a barrier was placed).</summary>
    public void Clear()
    {
        _pendingWrites.Clear();
        _pendingReads.Clear();
    }

    /// <summary>Records the accesses of <paramref name="launch"/> as pending.</summary>
    public void Add(KernelLaunch launch)
    {
        _pendingWrites.AddRange(launch.Writes);
        _pendingReads.AddRange(launch.Reads);
    }
}
