using Sharplight.Studio.Models;

namespace Sharplight.Studio.Services.Runtime;

/// <summary>
/// What <see cref="ProviderChoice.Automatic"/> consults: per-device knowledge of which provider is faster. The hook for the policy of
/// docs/cuda.md ("Provider selection": a catalog keyed by vendor and compute capability, flipped per device class once CUDA is measured
/// faster). Until such data exists, <see cref="NoProviderHints"/> keeps the default: Vulkan.
/// </summary>
internal interface IProviderPerformanceHints
{
    /// <summary>The provider that is known to be faster on a GPU, or null for the default (Vulkan).</summary>
    /// <param name="gpu">The GPU's entries (one per provider that lists it), with their features.</param>
    public string? FasterProvider(IReadOnlyList<ComputeDeviceOption> gpu);
}
