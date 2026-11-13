namespace Sharplight.Package;

/// <summary>The decoded header of an artifact section: descriptor and entry locations (absolute file offsets); no entry bytes.</summary>
internal sealed record ArtifactSetDirectory(
    PackageSection Section,
    IReadOnlyDictionary<string, string> Descriptor,
    IReadOnlyList<(string Key, long Offset, int Length)> Entries);
