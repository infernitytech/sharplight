using Vortice.Vulkan;

namespace Sharplight.Vulkan;

/// <summary>One vkAllocateMemory allocation: its memory type and size, for freeing and for the per-heap tally.</summary>
internal readonly record struct MemoryChunk(VkDeviceMemory Memory, uint TypeIndex, ulong Size);
