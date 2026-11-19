namespace Sharplight.Studio.Models;

/// <summary>The titles execution providers go by in the UI.</summary>
internal static class ProviderTitles
{
    /// <summary>The name of the Vulkan execution provider.</summary>
    public const string Vulkan = "vulkan";

    /// <summary>The name of the CUDA execution provider.</summary>
    public const string Cuda = "cuda";

    /// <summary>The title of a provider: <c>CUDA</c>, <c>Vulkan</c>, otherwise the name with a capital.</summary>
    /// <param name="provider">The provider's name.</param>
    public static string Of(string provider) => provider switch
    {
        Cuda => "CUDA",
        Vulkan => "Vulkan",
        "cpu" => "CPU",
        "metal" => "Metal",
        { Length: > 0 } => char.ToUpperInvariant(provider[0]) + provider[1..],
        _ => provider,
    };
}
