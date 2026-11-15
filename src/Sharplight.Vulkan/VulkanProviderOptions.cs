namespace Sharplight.Vulkan;

/// <summary>Options of the Vulkan execution provider (<c>services.AddVulkanProvider(options => ...)</c>).</summary>
public sealed class VulkanProviderOptions
{
    /// <summary>
    /// Enables the Khronos validation layer. The Vulkan instance is process-wide and created on first use, so only the options of
    /// the provider that creates it take effect. Default: on when the environment variable <c>SHARPLIGHT_VK_VALIDATION=1</c> is set.
    /// </summary>
    public bool EnableValidation { get; set; } = DiagnosticSwitches.Validation;
}
