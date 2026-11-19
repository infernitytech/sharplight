namespace Sharplight.Studio.Models;

/// <summary>Which execution provider runs a GPU that more than one provider can drive (an NVIDIA GPU: Vulkan and CUDA).</summary>
internal enum ProviderChoice
{
    /// <summary>Studio decides: Vulkan until per-device data says another provider is faster (see <c>IProviderPerformanceHints</c>).</summary>
    Automatic,

    /// <summary>Always the Vulkan provider.</summary>
    Vulkan,

    /// <summary>The CUDA provider; Vulkan when CUDA cannot be used on this machine.</summary>
    Cuda,
}
