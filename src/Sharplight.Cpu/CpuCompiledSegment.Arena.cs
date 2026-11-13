namespace Sharplight.Cpu;

// Liveness-based packing of all activation buffers into one arena allocation.
internal sealed unsafe partial class CpuCompiledSegment
{
    /// <summary>
    /// Assigns every arena buffer an offset: buffers are placed in order of first use, reusing (best fit) the
    /// space of buffers whose last use has passed; pinned buffers (inputs, outputs) live for the whole run.
    /// </summary>
    private void AllocateArena()
    {
        int end = _steps.Count;
        foreach (var b in _bufs)
        {
            if (b.Pinned)
            {
                b.First = 0;
                b.Last = end;
            }
            else if (b.First == int.MaxValue)
            {
                b.First = 0;
                b.Last = 0;
            }
        }

        var order = _bufs.OrderBy(b => b.First).ToList();
        var free = new List<(long Off, long Size)>();
        var active = new List<Buf>();
        long top = 0;
        int next = 0;
        for (int step = 0; step <= end; step++)
        {
            while (next < order.Count && order[next].First == step)
            {
                var buf = order[next++];
                PlaceBuffer(buf, free, ref top);
                active.Add(buf);
            }

            for (int i = active.Count - 1; i >= 0; i--)
            {
                var buf = active[i];
                if (buf.Last > step)
                {
                    continue;
                }

                active.RemoveAt(i);
                if (!buf.Pinned)
                {
                    ReleaseBuffer(buf, free);
                }
            }
        }

        _arenaBytes = top;
        _arena = Mem.Alloc(Math.Max(top, 64), true);
        foreach (var b in _bufs)
        {
            b.Ptr = _arena + b.ArenaOffset;
        }

        foreach (var r in _allRefs)
        {
            r.P = r.Buf.Ptr + r.Offset * r.ElemSize;
        }
    }

    /// <summary>Best-fit placement in the free list; otherwise grows the arena (extending a trailing free block when possible).</summary>
    private static void PlaceBuffer(Buf buf, List<(long Off, long Size)> free, ref long top)
    {
        int best = -1;
        for (int i = 0; i < free.Count; i++)
        {
            if (free[i].Size >= buf.Bytes && (best < 0 || free[i].Size < free[best].Size))
            {
                best = i;
            }
        }

        if (best >= 0)
        {
            buf.ArenaOffset = free[best].Off;
            long remaining = free[best].Size - buf.Bytes;
            if (remaining > 0)
            {
                free[best] = (free[best].Off + buf.Bytes, remaining);
            }
            else
            {
                free.RemoveAt(best);
            }
        }
        else if (free.Count > 0 && free[^1].Off + free[^1].Size == top)
        {
            buf.ArenaOffset = free[^1].Off;
            top = buf.ArenaOffset + buf.Bytes;
            free.RemoveAt(free.Count - 1);
        }
        else
        {
            buf.ArenaOffset = top;
            top += buf.Bytes;
        }
    }

    /// <summary>Returns the buffer's space to the offset-ordered free list, merging with adjacent free blocks.</summary>
    private static void ReleaseBuffer(Buf buf, List<(long Off, long Size)> free)
    {
        int pos = free.FindIndex(f => f.Off > buf.ArenaOffset);
        if (pos < 0)
        {
            pos = free.Count;
        }

        free.Insert(pos, (buf.ArenaOffset, buf.Bytes));
        if (pos + 1 < free.Count && free[pos].Off + free[pos].Size == free[pos + 1].Off)
        {
            free[pos] = (free[pos].Off, free[pos].Size + free[pos + 1].Size);
            free.RemoveAt(pos + 1);
        }

        if (pos > 0 && free[pos - 1].Off + free[pos - 1].Size == free[pos].Off)
        {
            free[pos - 1] = (free[pos - 1].Off, free[pos - 1].Size + free[pos].Size);
            free.RemoveAt(pos);
        }
    }
}
