namespace Sharplight.Package;

/// <summary>A provider (or one of its devices) <see cref="PackagePrecompiler"/> could not precompile for.</summary>
/// <param name="Provider">The execution provider.</param>
/// <param name="Device">The device, or empty when the whole provider was skipped.</param>
/// <param name="Reason">Why (unavailable, no devices, failed with ...).</param>
public sealed record SkippedProvider(string Provider, string Device, string Reason);
