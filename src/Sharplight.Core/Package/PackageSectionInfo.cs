namespace Sharplight.Package;

/// <summary>One section of a package, as listed in its section table.</summary>
/// <param name="Kind">Four-character kind: <c>DEFN</c>, <c>WIDX</c>, <c>WDAT</c>, <c>ARTS</c>, <c>KSET</c> (version 1), or the number of a kind this version does not know.</param>
/// <param name="Name">Section name: the artifact-set id for ARTS, the device target for KSET, empty otherwise.</param>
/// <param name="Offset">Absolute file offset.</param>
/// <param name="Length">Length in bytes.</param>
public sealed record PackageSectionInfo(string Kind, string Name, long Offset, long Length);
