using Sharplight.Runtime;

namespace Sharplight.Package;

/// <summary>What one precompilation stage produced for a device (a <see cref="CodeSet"/> without its bytes).</summary>
/// <param name="Descriptor">The set's descriptor, formatted (<see cref="ArtifactDescriptor.Format"/>).</param>
/// <param name="Kind">The artifact kind (e.g. <c>spirv</c>, <c>pipeline-binary</c>, <c>pipeline-cache</c>).</param>
/// <param name="Entries">Number of entries.</param>
/// <param name="Bytes">Sum of the entry sizes.</param>
/// <param name="ReusedEntries">Entries that were already stored in the package.</param>
/// <param name="SharedWith">The device whose identical set this device uses (same profile, or same device model and driver), or
/// <see langword="null"/> when the set was produced for this device.</param>
/// <param name="Elapsed">Time the stage took (zero when shared).</param>
public sealed record CodeSetSummary(string Descriptor, string Kind, int Entries, long Bytes, int ReusedEntries, string? SharedWith, TimeSpan Elapsed)
{
    /// <summary>Summarizes <paramref name="set"/>.</summary>
    public static CodeSetSummary Of(CodeSet set, string? sharedWith, TimeSpan elapsed)
    {
        ArgumentNullException.ThrowIfNull(set);
        return new CodeSetSummary(ArtifactDescriptor.Format(set.Descriptor), set.Kind, set.Entries.Count, set.TotalBytes, set.ReusedEntries, sharedWith, elapsed);
    }
}
