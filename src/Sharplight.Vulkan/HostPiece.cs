using Vortice.Vulkan;

namespace Sharplight.Vulkan;

/// <summary>
/// A piece of a host range (<see cref="HostMemoryImports.Split"/>): inside an import, the imported buffer and the piece's offset in it; in
/// a gap between imports, a null buffer (the piece is staged).
/// </summary>
internal readonly record struct HostPiece(nint Start, long Bytes, VkBuffer Buffer, ulong Offset);
