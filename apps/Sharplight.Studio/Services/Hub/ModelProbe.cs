using Sharplight.Studio.Services.Compatibility;

namespace Sharplight.Studio.Services.Hub;

/// <summary>What was read of a variant before downloading it.</summary>
/// <param name="Shape">The dimensions for the fit estimate, or null when they could not be read.</param>
/// <param name="Compatibility">The likely compatibility of this variant.</param>
/// <param name="ArchitectureCompatibility">
/// The verdict on the architecture alone, which the repository's other variants share until their own weight formats are read.
/// </param>
internal sealed record ModelProbe(ModelShape? Shape, ModelCompatibility Compatibility, ModelCompatibility ArchitectureCompatibility);
