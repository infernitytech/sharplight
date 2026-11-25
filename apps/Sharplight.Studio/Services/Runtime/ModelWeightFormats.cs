namespace Sharplight.Studio.Services.Runtime;

/// <summary>
/// The weight formats of a model that load options depend on, read from its definition without loading it
/// (<see cref="IModelRuntime.DescribeWeightFormatsAsync"/>).
/// </summary>
/// <param name="BlockScaledFp8">The model has block-scaled E4M3 (FP8) weight matrices.</param>
/// <param name="Nvfp4">The model has NVFP4 (block-scaled E2M1) weight matrices.</param>
internal sealed record ModelWeightFormats(bool BlockScaledFp8, bool Nvfp4)
{
    /// <summary>No such weights (or the model could not be read).</summary>
    public static ModelWeightFormats None { get; } = new(false, false);
}
