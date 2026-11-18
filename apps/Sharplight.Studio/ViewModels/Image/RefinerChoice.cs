using Sharplight.Studio.Services.Runtime;

namespace Sharplight.Studio.ViewModels.Image;

/// <summary>
/// A choice of the Image page's refiner: another resident image-generation model that finishes the chosen model's denoising (the SDXL refiner
/// after the SDXL base), or none.
/// </summary>
/// <param name="Label">What the list shows.</param>
/// <param name="Model">The refiner, or null for none.</param>
internal sealed record RefinerChoice(string Label, ILoadedModel? Model)
{
    /// <summary>No refiner.</summary>
    public static RefinerChoice None { get; } = new("None", null);

    /// <inheritdoc/>
    public override string ToString() => Label;
}
