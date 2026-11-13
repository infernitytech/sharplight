using Sharplight.Runtime;

namespace Sharplight.Package;

/// <summary>A stored artifact set: its descriptor and entry directory (sizes only, no entry bytes).</summary>
public sealed class ArtifactSetInfo
{
    internal ArtifactSetInfo(ArtifactSetDirectory directory)
    {
        Id = directory.Section.Name;
        IsLegacy = directory.Section.Kind == PackageSectionKind.KernelSet;
        Offset = directory.Section.Offset;
        Length = directory.Section.Length;
        Descriptor = directory.Descriptor;
        Entries = [.. directory.Entries.Select(e => new ArtifactEntryInfo(e.Key, e.Length))];
        TotalBytes = Entries.Sum(e => e.Size);
    }

    /// <summary>Section name: the descriptor id (<see cref="ArtifactDescriptor.Id"/>), or the device target of a version 1 kernel set.</summary>
    public string Id { get; }

    /// <summary>Whether this is a version 1 kernel set (KSET), read with tier <see cref="ArtifactTiers.Mixed"/>.</summary>
    public bool IsLegacy { get; }

    /// <summary>The descriptor (see <see cref="ArtifactDescriptorKeys"/>).</summary>
    public IReadOnlyDictionary<string, string> Descriptor { get; }

    /// <summary>Provider the set is for.</summary>
    public string Provider => ArtifactDescriptor.Get(Descriptor, ArtifactDescriptorKeys.Provider);

    /// <summary>Tier (<see cref="ArtifactTiers"/>).</summary>
    public string Tier => ArtifactDescriptor.Get(Descriptor, ArtifactDescriptorKeys.Tier);

    /// <summary>Artifact kind.</summary>
    public string Kind => ArtifactDescriptor.Get(Descriptor, ArtifactDescriptorKeys.Kind);

    /// <summary>Absolute file offset of the section.</summary>
    public long Offset { get; }

    /// <summary>Section length in bytes (entries, directory and padding).</summary>
    public long Length { get; }

    /// <summary>Entries in key order.</summary>
    public IReadOnlyList<ArtifactEntryInfo> Entries { get; }

    /// <summary>Number of entries.</summary>
    public int EntryCount => Entries.Count;

    /// <summary>Sum of the entry sizes.</summary>
    public long TotalBytes { get; }

    /// <inheritdoc/>
    public override string ToString() => ArtifactDescriptor.Format(Descriptor);
}
