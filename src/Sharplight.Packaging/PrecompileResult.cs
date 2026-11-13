namespace Sharplight.Packaging;

/// <summary>What <see cref="PackagePrecompiler"/> stored.</summary>
/// <param name="Device">Description of the device the package was compiled on (or for).</param>
/// <param name="StoredSets">Descriptors (formatted) of the artifact sets added to the package.</param>
/// <param name="UnchangedSets">Descriptors of recorded sets that were served entirely from sets already in the package.</param>
/// <param name="StoredBytes">Bytes of artifact entries added.</param>
/// <param name="PackageBytes">Size of the package afterwards.</param>
/// <param name="CompileTime">Time spent compiling.</param>
public sealed record PrecompileResult(
    string Device,
    IReadOnlyList<string> StoredSets,
    IReadOnlyList<string> UnchangedSets,
    long StoredBytes,
    long PackageBytes,
    TimeSpan CompileTime);
