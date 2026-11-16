namespace Sharplight.Package;

/// <summary>What <see cref="PackagePrecompiler"/> compiled and stored.</summary>
/// <param name="PackagePath">The package.</param>
/// <param name="Devices">Per device: the portable and device-specific stage.</param>
/// <param name="Skipped">Providers and devices that were skipped, with the reason.</param>
/// <param name="StoredSets">Descriptors (formatted) of the artifact sets written into the package.</param>
/// <param name="UnchangedSets">Descriptors of produced sets the package already held unchanged.</param>
/// <param name="StoredBytes">Bytes of artifact entries written.</param>
/// <param name="PackageBytes">Size of the package afterwards.</param>
/// <param name="Elapsed">Total time.</param>
public sealed record PackagePrecompileReport(
    string PackagePath,
    IReadOnlyList<DevicePrecompileResult> Devices,
    IReadOnlyList<SkippedProvider> Skipped,
    IReadOnlyList<string> StoredSets,
    IReadOnlyList<string> UnchangedSets,
    long StoredBytes,
    long PackageBytes,
    TimeSpan Elapsed);
