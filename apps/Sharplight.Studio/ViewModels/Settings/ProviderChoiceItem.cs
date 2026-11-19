using Sharplight.Studio.Models;

namespace Sharplight.Studio.ViewModels.Settings;

/// <summary>An entry of a GPU's provider picker.</summary>
/// <param name="Value">The choice.</param>
/// <param name="Title">Name in the picker.</param>
internal sealed record ProviderChoiceItem(ProviderChoice Value, string Title)
{
    /// <summary>Studio decides (Vulkan until per-device data says otherwise).</summary>
    public static ProviderChoiceItem Automatic { get; } = new(ProviderChoice.Automatic, "Automatic");

    /// <summary>Vulkan.</summary>
    public static ProviderChoiceItem Vulkan { get; } = new(ProviderChoice.Vulkan, "Vulkan");

    /// <summary>CUDA (NVIDIA GPUs).</summary>
    public static ProviderChoiceItem Cuda { get; } = new(ProviderChoice.Cuda, "CUDA");

    /// <summary>The choice's item.</summary>
    /// <param name="choice">A choice.</param>
    public static ProviderChoiceItem For(ProviderChoice choice) => choice switch
    {
        ProviderChoice.Vulkan => Vulkan,
        ProviderChoice.Cuda => Cuda,
        _ => Automatic,
    };
}
