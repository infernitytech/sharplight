namespace Sharplight.Distributed.Worker;

/// <summary>
/// The boundary values neighbouring workers sent to this worker, by tag (one tag per forwarded segment input) and execution number. An
/// execution takes the value of its own number; values of earlier numbers (left over from an aborted execution) are dropped.
/// </summary>
internal sealed class PeerMailbox
{
    private readonly Dictionary<long, Slot> _slots = [];
    private readonly Lock _lock = new();
    private long _nextTag;

    /// <summary>A new tag for a forwarded input.</summary>
    public long Register()
    {
        lock (_lock)
        {
            long tag = ++_nextTag;
            _slots[tag] = new Slot();
            return tag;
        }
    }

    /// <summary>Forgets a tag and the values queued under it.</summary>
    public void Unregister(long tag)
    {
        lock (_lock)
        {
            _slots.Remove(tag);
        }
    }

    /// <summary>Queues a value (or a producer's failure, <paramref name="data"/> null) for execution <paramref name="sequence"/>.</summary>
    /// <param name="tag">The tag the consumer registered.</param>
    /// <param name="sequence">The producer's execution number.</param>
    /// <param name="data">The value, or null when the producer failed.</param>
    /// <param name="error">Why the producer failed.</param>
    /// <param name="source">The link the value came over (null: this worker).</param>
    public void Post(long tag, long sequence, byte[]? data, string? error, object? source)
    {
        lock (_lock)
        {
            if (!_slots.TryGetValue(tag, out var slot))
            {
                return;
            }

            slot.Source = source;
            slot.Values.Enqueue(new Entry(sequence, data, error));
        }

        PulseWaiters(tag);
    }

    /// <summary>Fails the waits for tags last fed over a link that closed.</summary>
    public void SourceClosed(object source, string reason)
    {
        List<long> tags;
        lock (_lock)
        {
            tags = [.. _slots.Where(kv => ReferenceEquals(kv.Value.Source, source)).Select(kv => kv.Key)];
            foreach (var tag in tags)
            {
                _slots[tag].Closed = reason;
            }
        }

        foreach (var tag in tags)
        {
            PulseWaiters(tag);
        }
    }

    /// <summary>Waits for the value of execution <paramref name="sequence"/> under <paramref name="tag"/>.</summary>
    /// <exception cref="IOException">The producer failed, its link closed, a value is missing, or the wait timed out.</exception>
    /// <exception cref="OperationCanceledException">The coordinator aborted the execution.</exception>
    public byte[] Take(long tag, long sequence, TimeSpan timeout, CancellationToken cancellationToken)
    {
        Slot slot;
        lock (_lock)
        {
            slot = _slots.TryGetValue(tag, out var s) ? s : throw new InvalidOperationException($"No forwarded input with tag {tag}.");
        }

        var deadline = DateTime.UtcNow + timeout;
        using var registration = cancellationToken.Register(() => PulseWaiters(tag));
        lock (slot)
        {
            while (true)
            {
                cancellationToken.ThrowIfCancellationRequested();
                Entry? entry = null;
                lock (_lock)
                {
                    while (slot.Values.TryPeek(out var head) && head.Sequence < sequence)
                    {
                        slot.Values.Dequeue();
                    }

                    if (slot.Values.TryPeek(out var next))
                    {
                        entry = next;
                        if (next.Sequence == sequence)
                        {
                            slot.Values.Dequeue();
                        }
                    }
                    else if (slot.Closed is { } closed)
                    {
                        throw new IOException($"The neighbour sending this segment's input disconnected: {closed}");
                    }
                }

                if (entry is { } found)
                {
                    if (found.Sequence != sequence)
                    {
                        throw new IOException($"The neighbour sent the input of execution {found.Sequence}; execution {sequence} is waiting for its own.");
                    }

                    return found.Data ?? throw new IOException($"The neighbour producing this segment's input failed: {found.Error}");
                }

                var remaining = deadline - DateTime.UtcNow;
                if (remaining <= TimeSpan.Zero)
                {
                    throw new IOException($"The neighbour sent no input for execution {sequence} within {timeout.TotalSeconds:F0} s.");
                }

                Monitor.Wait(slot, remaining < TimeSpan.FromSeconds(1) ? remaining : TimeSpan.FromSeconds(1));
            }
        }
    }

    private void PulseWaiters(long tag)
    {
        Slot? slot;
        lock (_lock)
        {
            _slots.TryGetValue(tag, out slot);
        }

        if (slot is not null)
        {
            lock (slot)
            {
                Monitor.PulseAll(slot);
            }
        }
    }

    private readonly record struct Entry(long Sequence, byte[]? Data, string? Error);

    private sealed class Slot
    {
        public Queue<Entry> Values { get; } = new();

        public object? Source { get; set; }

        public string? Closed { get; set; }
    }
}
