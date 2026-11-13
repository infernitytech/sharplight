namespace Sharplight.Llm.Serving;

/// <summary>Scheduler view of one decoder slot: the tokens its context holds and the sequence using it.</summary>
internal sealed class EngineSlot(int index)
{
    /// <summary>Decoder slot index.</summary>
    public int Index { get; } = index;
    /// <summary>The tokens in the slot's context (kept after a request ends, for prefix reuse).</summary>
    public List<int> Tokens { get; } = [];
    /// <summary>Logical time of the last release (least recently used slots are evicted first).</summary>
    public long LastUsed { get; set; }
    /// <summary>The sequence running in the slot, or null when idle.</summary>
    public EngineSequence? Active { get; set; }
}
