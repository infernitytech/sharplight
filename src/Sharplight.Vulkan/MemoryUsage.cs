namespace Sharplight.Vulkan;

internal enum MemoryUsage
{
    /// <summary>Device local, GPU only.</summary>
    Device,
    /// <summary>Host writes, device reads (prefers device-local host-visible "ReBAR" memory).</summary>
    Upload,
    /// <summary>Device writes, host reads (prefers host-cached memory).</summary>
    Readback,
}
