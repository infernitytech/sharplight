namespace Sharplight.Package;

/// <summary>Kinds of package sections (the <c>u32 kind</c> of a section table entry).</summary>
internal enum PackageSectionKind : uint
{
    /// <summary>DEFN: Sharplight IR graphs, states, properties, metadata.</summary>
    Definition = 1,
    /// <summary>WIDX: weight index (name, type, shape, offset, length).</summary>
    WeightIndex = 2,
    /// <summary>WDAT: weight bytes, 64 KiB aligned.</summary>
    WeightData = 3,
    /// <summary>KSET: version 1 kernel set of one device target (read as a legacy artifact set).</summary>
    KernelSet = 4,
    /// <summary>ARTS: one artifact set (descriptor and entries), version 2.</summary>
    ArtifactSet = 5,
}
