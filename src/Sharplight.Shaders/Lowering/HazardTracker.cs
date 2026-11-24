namespace Sharplight.Shaders.Lowering;

/// <summary>
/// Decides where a command buffer needs memory barriers: a launch needs one before it when it reads data written, or
/// writes data read or written, by a launch since the previous barrier.
/// </summary>
internal sealed class HazardTracker
{
    /// <summary>Launches examined ahead of the first deferred one when filling a barrier batch (see <see cref="Order(IReadOnlyList{KernelLaunch}, Bindings, bool)"/>).</summary>
    private const int Lookahead = 32;

    private readonly List<(BufferRange Range, object? Group)> _pendingWrites = [];
    private readonly List<BufferRange> _pendingReads = [];

    /// <summary>True when <paramref name="launch"/> conflicts with a launch recorded since the last barrier.</summary>
    public bool Conflicts(KernelLaunch launch)
    {
        // Indexed loops: LINQ over the access lists allocates closures and enumerators per check.
        for (int r = 0; r < launch.Reads.Count; r++)
        {
            var read = launch.Reads[r];
            foreach (var (range, _) in _pendingWrites)
            {
                if (range.Overlaps(read))
                {
                    return true;
                }
            }
        }

        for (int w = 0; w < launch.Writes.Count; w++)
        {
            var write = launch.Writes[w];
            foreach (var (range, group) in _pendingWrites)
            {
                if (range.Overlaps(write) && (group is null || !ReferenceEquals(group, launch.WriteGroup)))
                {
                    return true;
                }
            }

            foreach (var range in _pendingReads)
            {
                if (range.Overlaps(write))
                {
                    return true;
                }
            }
        }

        return false;
    }

    /// <summary>Forgets all pending accesses (a barrier was placed).</summary>
    public void Clear()
    {
        _pendingWrites.Clear();
        _pendingReads.Clear();
    }

    /// <summary>Records the accesses of <paramref name="launch"/> as pending.</summary>
    public void Add(KernelLaunch launch)
    {
        _pendingWrites.AddRange(launch.Writes.Select(w => (w, launch.WriteGroup)));
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

        return Order(pending, reorder);
    }

    /// <summary>
    /// <see cref="Order(IReadOnlyList{KernelLaunch}, Bindings, bool)"/> with the barrier placement memoized in
    /// <paramref name="cache"/> by the set of launches that run: the accesses of a launch do not depend on the bindings, so
    /// bindings that select the same launches (a new detection count, say) reuse the order and only re-select variants.
    /// <paramref name="scope"/> tells apart the launch lists sharing one cache (the program index of the first launch).
    /// </summary>
    public static List<(KernelLaunch Launch, int Index, KernelVariant Variant, bool BarrierBefore)> Order(IReadOnlyList<KernelLaunch> launches, Bindings bindings, bool reorder,
        Dictionary<string, (int Index, bool BarrierBefore)[]> cache, int scope)
    {
        var selected = new KernelVariant?[launches.Count];
        var key = new char[launches.Count + 1];
        key[0] = reorder ? 'r' : 'p';
        for (int i = 0; i < launches.Count; i++)
        {
            selected[i] = launches[i].Select(bindings);
            key[i + 1] = selected[i] is null ? '0' : '1';
        }

        string k = scope.ToString(System.Globalization.CultureInfo.InvariantCulture) + new string(key);
        if (!cache.TryGetValue(k, out var placement))
        {
            var pending = new List<(KernelLaunch Launch, int Index, KernelVariant Variant)>();
            for (int i = 0; i < launches.Count; i++)
            {
                if (selected[i] is { } variant)
                {
                    pending.Add((launches[i], i, variant));
                }
            }

            if (cache.Count >= MaxCachedOrders)
            {
                cache.Clear();
            }

            placement = [.. Order(pending, reorder).Select(o => (o.Index, o.BarrierBefore))];
            cache[k] = placement;
        }

        return [.. placement.Select(o => (launches[o.Index], o.Index, selected[o.Index]!, o.BarrierBefore))];
    }

    /// <summary>Most launch sets whose order a cache passed to <see cref="Order(IReadOnlyList{KernelLaunch}, Bindings, bool, Dictionary{string, ValueTuple{int, bool}[]}, int)"/> keeps.</summary>
    private const int MaxCachedOrders = 64;

    private static List<(KernelLaunch Launch, int Index, KernelVariant Variant, bool BarrierBefore)> Order(List<(KernelLaunch Launch, int Index, KernelVariant Variant)> pending, bool reorder)
    {
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

        // Positions in pending (which stays in program order); conflicts between two launches are computed once.
        var conflicts = new LaunchConflicts([.. pending.Select(p => p.Launch)]);
        var remaining = new List<int>(Enumerable.Range(0, pending.Count));
        var next = new List<int>(pending.Count);
        var taken = new List<int>();
        var deferred = new List<int>();
        bool barrierBefore = false;
        while (remaining.Count > 0)
        {
            // One barrier batch: take launches in program order while they fit; a launch that conflicts with the batch is
            // deferred, and so is every later one that conflicts with a deferred launch.
            taken.Clear();
            deferred.Clear();
            next.Clear();
            for (int i = 0; i < remaining.Count; i++)
            {
                int position = remaining[i];
                if (deferred.Count >= Lookahead)
                {
                    next.AddRange(remaining.GetRange(i, remaining.Count - i));
                    break;
                }

                // A deferred launch's conflict: the position reads what it writes, or writes what it reads or writes.
                if (taken.Count == 0 || (!conflicts.Any(position, taken) && !conflicts.Any(position, deferred)))
                {
                    var item = pending[position];
                    order.Add((item.Launch, item.Index, item.Variant, barrierBefore && taken.Count == 0));
                    taken.Add(position);
                }
                else
                {
                    deferred.Add(position);
                    next.Add(position);
                }
            }

            (remaining, next) = (next, remaining);
            barrierBefore = true;
        }

        return order;
    }
}
