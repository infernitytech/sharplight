using Vortice.Vulkan;

namespace Sharplight.Vulkan;

internal sealed unsafe class MemoryBlock
{
    public required VkDeviceMemory Memory { get; init; }
    public required VkBuffer Buffer { get; init; }
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
