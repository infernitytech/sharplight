namespace Sharplight.ModelFeed;

/// <summary>Memory needed on one device kind.</summary>
public sealed record MemoryRequirement
{
    /// <summary>Device kind (<c>gpu</c>, <c>cpu</c>).</summary>
    public required string Device { get; init; }

    /// <summary>Weight bytes.</summary>
    public long Weights { get; init; }

    /// <summary>Context memory per slot at a reference length.</summary>
    public SlotContextMemory? PerSlotContext { get; init; }

    /// <summary>Fixed overhead bytes (activations, scratch).</summary>
    public long Overhead { get; init; }

    /// <summary>Bytes for <paramref name="slots"/> slots at the reference context length.</summary>
    /// <param name="slots">Number of slots.</param>
    /// <returns>Total bytes.</returns>
    public long TotalFor(int slots) => Weights + Overhead + (PerSlotContext?.Bytes ?? 0) * Math.Max(0, slots);
}
