namespace Sharplight.Package;

/// <summary>One entry of the section table: kind, name (the artifact-set id for ARTS, the device target for KSET), absolute offset and length.</summary>
internal readonly record struct PackageSection(PackageSectionKind Kind, string Name, long Offset, long Length)
{
    /// <summary>Whether the section holds precompiled artifacts (ARTS or legacy KSET).</summary>
    public bool IsArtifactSet => Kind is PackageSectionKind.ArtifactSet or PackageSectionKind.KernelSet;

    /// <summary>Four-character tag of the kind (<c>DEFN</c>, <c>WIDX</c>, <c>WDAT</c>, <c>KSET</c>, <c>ARTS</c>), or the number of an unknown kind.</summary>
    public string Tag => Kind switch
    {
        PackageSectionKind.Definition => "DEFN",
        PackageSectionKind.WeightIndex => "WIDX",
        PackageSectionKind.WeightData => "WDAT",
        PackageSectionKind.KernelSet => "KSET",
        PackageSectionKind.ArtifactSet => "ARTS",
        _ => ((uint)Kind).ToString(System.Globalization.CultureInfo.InvariantCulture),
    };
}
