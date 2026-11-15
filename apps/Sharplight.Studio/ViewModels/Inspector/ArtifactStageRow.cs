namespace Sharplight.Studio.ViewModels.Inspector;

/// <summary>One precompilation stage stored for a provider: portable code for a device profile, or device code for a device and driver.</summary>
/// <param name="Provider">The execution provider.</param>
/// <param name="Stage">"Portable code", "Device code" or "Legacy kernels".</param>
/// <param name="Target">What it was built for (profile, or device and driver).</param>
/// <param name="Detail">Kind, entries and size.</param>
internal sealed record ArtifactStageRow(string Provider, string Stage, string Target, string Detail);
