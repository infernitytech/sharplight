using Vortice.Vulkan;

namespace Sharplight.Vulkan;

/// <summary>One VkBuffer and the memory behind it: one allocation, or several bound sparsely when the buffer is larger than one allocation may be.</summary>
internal sealed unsafe class MemoryBlock
{
    public required VkBuffer Buffer { get; init; }
    /// <summary>The allocations backing the buffer, in buffer order (one unless the buffer is sparse).</summary>
    public required IReadOnlyList<MemoryChunk> Memory { get; init; }
    public required ulong Size { get; init; }
    public required ulong Address { get; init; }
    public byte* Mapped { get; init; }
    public bool Dedicated { get; init; }
    public bool Coherent { get; init; }
    public MemoryUsage Usage { get; init; }
    /// <summary>Bytes counted against the device-local budget (0 for host-only memory).</summary>
    public long Tracked { get; init; }
    public readonly SortedList<ulong, ulong> Free = [];   // offset → size
}
