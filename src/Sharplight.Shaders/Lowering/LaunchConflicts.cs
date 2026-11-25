namespace Sharplight.Shaders.Lowering;

/// <summary>
/// Pairwise conflicts between the launches of one recording, computed on first use and remembered: whether a later launch reads what
/// an earlier one writes, or writes what it reads or writes (outside a shared write group), which is what
/// <see cref="HazardTracker.Conflicts"/> answers for the union of a batch's accesses. Filling barrier batches asks about the same
/// pairs again and again (a launch deferred past a batch is examined against the next batch's deferred launches too), so each pair's
/// answer is two bits of a triangular matrix (up to <see cref="MaxMatrixLaunches"/> launches; a dictionary beyond).
/// </summary>
internal sealed class LaunchConflicts
{
    /// <summary>Most launches whose answers live in the bit matrix (two bits per pair: about 2 MB at this size).</summary>
    private const int MaxMatrixLaunches = 4096;

    private readonly BufferRange[][] _reads;
    private readonly BufferRange[][] _writes;
    private readonly object?[] _groups;
    private readonly ulong[]? _known;
    private readonly ulong[]? _conflicts;
    private readonly Dictionary<long, bool>? _answers;

    /// <summary>Conflicts among <paramref name="launches"/>, by their position in the list.</summary>
    public LaunchConflicts(IReadOnlyList<KernelLaunch> launches)
    {
        // Arrays, not the launches' lists: indexing them allocates no enumerators (several per pair through the interface).
        _reads = [.. launches.Select(l => l.Reads.ToArray())];
        _writes = [.. launches.Select(l => l.Writes.ToArray())];
        _groups = [.. launches.Select(l => l.WriteGroup)];
        long n = launches.Count;
        if (n <= MaxMatrixLaunches)
        {
            long words = (n * (n - 1) / 2 + 63) / 64;
            _known = new ulong[words];
            _conflicts = new ulong[words];
        }
        else
        {
            _answers = [];
        }
    }

    /// <summary>True when the launch at <paramref name="later"/> conflicts with any launch at <paramref name="earlier"/> (all before it).</summary>
    public bool Any(int later, List<int> earlier)
    {
        foreach (int position in earlier)
        {
            if (Conflict(later, position))
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>True when the launch at <paramref name="later"/> must not share a barrier batch with the one at <paramref name="earlier"/>.</summary>
    private bool Conflict(int later, int earlier)
    {
        long pair = (long)later * (later - 1) / 2 + earlier;
        if (_answers is not null)
        {
            if (!_answers.TryGetValue(pair, out bool answer))
            {
                answer = Compute(later, earlier);
                _answers[pair] = answer;
            }

            return answer;
        }

        int word = (int)(pair >> 6);
        ulong bit = 1UL << (int)(pair & 63);
        if ((_known![word] & bit) == 0)
        {
            _known[word] |= bit;
            if (Compute(later, earlier))
            {
                _conflicts![word] |= bit;
            }
        }

        return (_conflicts![word] & bit) != 0;
    }

    /// <summary>The accesses of launch <paramref name="later"/> against those of <paramref name="earlier"/>, as <see cref="HazardTracker.Conflicts"/> checks them.</summary>
    private bool Compute(int later, int earlier)
    {
        var reads = _reads[later];
        var writes = _writes[later];
        var earlierReads = _reads[earlier];
        var earlierWrites = _writes[earlier];
        for (int r = 0; r < reads.Length; r++)
        {
            for (int w = 0; w < earlierWrites.Length; w++)
            {
                if (earlierWrites[w].Overlaps(reads[r]))
                {
                    return true;
                }
            }
        }

        bool sharedGroup = _groups[earlier] is not null && ReferenceEquals(_groups[earlier], _groups[later]);
        for (int w = 0; w < writes.Length; w++)
        {
            if (!sharedGroup)
            {
                for (int o = 0; o < earlierWrites.Length; o++)
                {
                    if (earlierWrites[o].Overlaps(writes[w]))
                    {
                        return true;
                    }
                }
            }

            for (int r = 0; r < earlierReads.Length; r++)
            {
                if (earlierReads[r].Overlaps(writes[w]))
                {
                    return true;
                }
            }
        }

        return false;
    }
}
