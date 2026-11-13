using System.Diagnostics;

namespace Sharplight.Llm.Serving;

/// <summary>
/// The engine's wait queue. <see cref="TakeNext"/> picks the highest <see cref="GenerationRequest.Priority"/>; among equal
/// priorities the tenant with the least recent usage (tokens charged with <see cref="Charge"/>, decaying with a half-life)
/// goes first, then the earliest submission. Not thread-safe: the engine guards it with its lock.
/// </summary>
internal sealed class RequestQueue(TimeSpan usageHalfLife)
{
    private readonly List<EngineSequence> _items = [];
    private readonly Dictionary<string, (double Tokens, long Ticks)> _usage = [];
    private readonly double _decayPerTick = Math.Log(2) / Math.Max(1, usageHalfLife.TotalSeconds * Stopwatch.Frequency);

    public int Count => _items.Count;

    public IReadOnlyList<EngineSequence> Items => _items;

    public void Add(EngineSequence seq) => _items.Add(seq);

    public bool Remove(EngineSequence seq) => _items.Remove(seq);

    public void Clear() => _items.Clear();

    /// <summary>Records tokens a tenant consumed (prompt rows computed plus generated tokens).</summary>
    public void Charge(string? tenant, int tokens)
    {
        if (tokens <= 0)
        {
            return;
        }

        string key = tenant ?? "";
        long now = Stopwatch.GetTimestamp();
        _usage[key] = (Usage(key, now) + tokens, now);
    }

    /// <summary>Removes and returns the request to admit next, or null when empty.</summary>
    public EngineSequence? TakeNext()
    {
        if (_items.Count == 0)
        {
            return null;
        }

        long now = Stopwatch.GetTimestamp();
        int best = 0;
        double bestUsage = Usage(_items[0].Request.Tenant ?? "", now);
        for (int i = 1; i < _items.Count; i++)
        {
            var candidate = _items[i].Request;
            var current = _items[best].Request;
            if (candidate.Priority < current.Priority)
            {
                continue;
            }

            double usage = Usage(candidate.Tenant ?? "", now);
            // Strictly better only: earlier submissions win ties.
            if (candidate.Priority > current.Priority || usage < bestUsage)
            {
                best = i;
                bestUsage = usage;
            }
        }

        var seq = _items[best];
        _items.RemoveAt(best);
        return seq;
    }

    private double Usage(string tenant, long now)
    {
        if (!_usage.TryGetValue(tenant, out var u))
        {
            return 0;
        }

        return u.Tokens * Math.Exp(-_decayPerTick * (now - u.Ticks));
    }
}
