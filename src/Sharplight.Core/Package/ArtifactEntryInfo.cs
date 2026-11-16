namespace Sharplight.Package;

/// <summary>One entry of a stored artifact set.</summary>
/// <param name="Key">Entry key (e.g. <c>kernel/&lt;content hash&gt;</c>, <c>vulkan/pipeline_cache</c>).</param>
/// <param name="Size">Size in bytes.</param>
public sealed record ArtifactEntryInfo(string Key, long Size)
{
    /// <summary>Absolute file offset of the entry bytes.</summary>
    public long Offset { get; init; }
}
