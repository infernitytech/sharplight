using Vortice.Vulkan;

namespace Sharplight.Vulkan;

/// <summary>A sub-allocation addressed by device address.</summary>
internal sealed unsafe class Allocation
{
    public required VkBuffer Buffer { get; init; }
    public required ulong Offset { get; init; }
    public required ulong Size { get; init; }
    public required ulong Address { get; init; }
    /// <summary>Host pointer for host-visible memory, else null.</summary>
    public byte* Mapped { get; init; }
    internal MemoryBlock? Block { get; init; }
    internal bool Coherent { get; init; } = true;
}
