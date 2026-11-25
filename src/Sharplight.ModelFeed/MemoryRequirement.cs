namespace Sharplight.ModelFeed;

/// <summary>
/// Memory needed on one device kind. A slot (one concurrent sequence) at <c>c</c> tokens of context needs
/// <see cref="StateBytesPerSlot"/> + <see cref="KvBytesPerToken"/> · c; the whole load needs <see cref="Weights"/> +
/// <see cref="Overhead"/> + the slots (see <see cref="TotalFor(int, int)"/>).
/// </summary>
public sealed record MemoryRequirement
{
    /// <summary>Device kind (<c>gpu</c>: device-local memory of an accelerator, <c>cpu</c>: system memory).</summary>
    public required string Device { get; init; }

    /// <summary>Weight bytes.</summary>
    public long Weights { get; init; }

    /// <summary>Context memory per slot at the default context length (<see cref="SlotContextMemory.Tokens"/>).</summary>
    public SlotContextMemory? PerSlotContext { get; init; }

    /// <summary>Context memory per slot at the model's maximum context length, or <see langword="null"/> when unknown.</summary>
    public SlotContextMemory? PerSlotContextMax { get; init; }

    /// <summary>KV cache bytes per token of one slot over all layers (16-bit cache), or <see langword="null"/> when unknown.</summary>
    public long? KvBytesPerToken { get; init; }

    /// <summary>Recurrent-state bytes per slot that do not grow with the context (linear-attention and SSM layers), or <see langword="null"/> when unknown.</summary>
    public long? StateBytesPerSlot { get; init; }

    /// <summary>Fixed overhead bytes (activations, scratch, runtime).</summary>
    public long Overhead { get; init; }

    /// <summary>
    /// Bytes of <see cref="Weights"/> that are routed experts of a mixture-of-experts model (expert banks of which a token selects only a few), or
    /// <see langword="null"/> when unknown or the model has none. These are the weights the engine can leave on disk and read on demand
    /// (<c>TaskLoadOptions.ExpertsOnDisk</c>) when the model is larger than the memory of the host; the rest (<see cref="Weights"/> minus these)
    /// must fit memory.
    /// </summary>
    public long? ExpertWeights { get; init; }

    /// <summary>
    /// Bytes of the routed experts one token selects over every layer (each expert product reads the selected experts of its bank), or
    /// <see langword="null"/> when unknown: what one decode step reads from disk when none of them is cached.
    /// </summary>
    public long? ExpertBytesPerToken { get; init; }

    /// <summary>Bytes for <paramref name="slots"/> slots at the default context length.</summary>
    /// <param name="slots">Number of slots.</param>
    /// <returns>Total bytes.</returns>
    public long TotalFor(int slots) => Weights + Overhead + (PerSlotContext?.Bytes ?? 0) * Math.Max(0, slots);

    /// <summary>
    /// Bytes for <paramref name="slots"/> slots at <paramref name="contextTokens"/> tokens each: exact from <see cref="KvBytesPerToken"/>
    /// and <see cref="StateBytesPerSlot"/> when known, else <see cref="PerSlotContext"/> scaled linearly.
    /// </summary>
    /// <param name="slots">Number of slots.</param>
    /// <param name="contextTokens">Context tokens per slot.</param>
    /// <returns>Total bytes.</returns>
    public long TotalFor(int slots, int contextTokens) => Weights + Overhead + ContextBytes(contextTokens) * Math.Max(0, slots);

    /// <summary>Context memory of one slot at <paramref name="contextTokens"/> tokens.</summary>
    /// <param name="contextTokens">Context tokens.</param>
    /// <returns>Bytes; 0 when nothing is known.</returns>
    public long ContextBytes(int contextTokens)
    {
        long tokens = Math.Max(0, contextTokens);
        if (KvBytesPerToken is { } perToken)
        {
            return (StateBytesPerSlot ?? 0) + perToken * tokens;
        }

        if (PerSlotContext is { Tokens: > 0 } reference)
        {
            return (long)(reference.Bytes * (tokens / (double)reference.Tokens));
        }

        return 0;
    }
}
