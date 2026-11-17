namespace Sharplight.Metal;

/// <summary>Options of the Metal execution provider (<c>services.AddMetalProvider(options => ...)</c>).</summary>
public sealed class MetalProviderOptions
{
    /// <summary>
    /// Lists only GPUs with unified memory (Apple silicon). Default: <see langword="true"/>; the kernels and buffer placement are tuned for
    /// shared storage, and on Intel Macs with discrete GPUs the Vulkan provider (MoltenVK) remains the better choice.
    /// </summary>
    public bool UnifiedMemoryOnly { get; set; } = true;
}
