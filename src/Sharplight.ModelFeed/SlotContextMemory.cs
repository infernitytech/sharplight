namespace Sharplight.ModelFeed;

/// <summary>Context (KV cache) memory per slot at a reference length.</summary>
public sealed record SlotContextMemory
{
    /// <summary>Reference context length in tokens.</summary>
    public int Tokens { get; init; }

    /// <summary>Bytes per slot at that length.</summary>
    public long Bytes { get; init; }
}
