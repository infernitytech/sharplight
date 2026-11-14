using Sharplight.Shaders.Lowering;

namespace Sharplight.Vulkan;

/// <summary>
/// Decides where a command buffer needs memory barriers: a launch needs one before it when it reads data written, or
/// writes data read or written, by a launch since the previous barrier.
/// </summary>
internal sealed class HazardTracker
{
    /// <summary>Launches examined ahead of the first deferred one when filling a barrier batch (see <see cref="Order"/>).</summary>
    private const int Lookahead = 32;

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

    /// <summary>
    /// Recording order of the launches that run for <paramref name="bindings"/>, with the barriers they need. With
    /// <paramref name="reorder"/>, every barrier batch is filled with later launches that conflict neither with the batch
    /// nor with any earlier launch deferred past it (within <see cref="Lookahead"/> launches), so independent chains such as
    /// the activation conversion, GEMM and reduction of sibling matmuls share barriers instead of each paying its own.
    /// Moving a launch ahead of launches it does not conflict with keeps every read-after-write, write-after-read and
    /// write-after-write order of the program, so results are unchanged. Without it the program order is kept.
    /// </summary>
    /// <returns>The launch, its program index, the selected variant and whether a barrier precedes it.</returns>
    public static List<(KernelLaunch Launch, int Index, KernelVariant Variant, bool BarrierBefore)> Order(IReadOnlyList<KernelLaunch> launches, Bindings bindings, bool reorder)
    {
        var pending = new List<(KernelLaunch Launch, int Index, KernelVariant Variant)>();
        for (int i = 0; i < launches.Count; i++)
        {
            if (launches[i].Select(bindings) is { } variant)
            {
                pending.Add((launches[i], i, variant));
            }
        }

        var order = new List<(KernelLaunch, int, KernelVariant, bool)>(pending.Count);
        var batch = new HazardTracker();
        if (!reorder)
        {
            foreach (var (launch, index, variant) in pending)
            {
                bool barrier = batch.Conflicts(launch);
                if (barrier)
                {
                    batch.Clear();
                }

                batch.Add(launch);
                order.Add((launch, index, variant, barrier));
            }

            return order;
        }

        var deferred = new HazardTracker();
        var next = new List<(KernelLaunch Launch, int Index, KernelVariant Variant)>(pending.Count);
        bool barrierBefore = false;
        while (pending.Count > 0)
        {
            // One barrier batch: take launches in program order while they fit; a launch that conflicts with the batch is
            // deferred, and so is every later one that conflicts with a deferred launch.
            batch.Clear();
            deferred.Clear();
            next.Clear();
            int examined = 0;
            bool first = true;
            for (int i = 0; i < pending.Count; i++)
            {
                var item = pending[i];
                if (examined >= Lookahead)
                {
                    next.AddRange(pending.Skip(i));
                    break;
                }

                // deferred.Conflicts: item reads what a deferred launch writes, or writes what one reads or writes.
                if (first || (!batch.Conflicts(item.Launch) && !deferred.Conflicts(item.Launch)))
                {
                    order.Add((item.Launch, item.Index, item.Variant, barrierBefore && first));
                    batch.Add(item.Launch);
                    first = false;
                }
                else
                {
                    deferred.Add(item.Launch);
                    next.Add(item);
                    examined++;
                }
            }

            (pending, next) = (next, pending);
            barrierBefore = true;
        }

        return order;
    }
}
