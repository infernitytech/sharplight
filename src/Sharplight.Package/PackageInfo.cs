namespace Sharplight.Package;

/// <summary>
/// Everything about a package file short of its weight bytes and artifact entry bytes: produced by
/// <see cref="ModelPackageReader.Open"/>. A snapshot: the file is closed when it is returned. Thread-safe.
/// </summary>
public sealed class PackageInfo
{
    private readonly Lazy<DefinitionInfo> _definition;

    internal PackageInfo(string path, long fileSize, int version, IReadOnlyList<PackageSectionInfo> sections, IReadOnlyList<PackageWeightInfo> weights,
        IReadOnlyList<ArtifactSetInfo> artifactSets, long compactedSize, Func<DefinitionInfo> definition)
    {
        Path = path;
        FileSize = fileSize;
        Version = version;
        Sections = sections;
        Weights = weights;
        WeightBytes = weights.Sum(w => w.ByteSize);
        ArtifactSets = artifactSets;
        ReclaimableBytes = Math.Max(0, fileSize - compactedSize);
        _definition = new(definition);
    }

    /// <summary>The file that was read.</summary>
    public string Path { get; }

    /// <summary>File size in bytes.</summary>
    public long FileSize { get; }

    /// <summary>Format version of the file (1 or 2, see <see cref="ModelPackage.Version"/>).</summary>
    public int Version { get; }

    /// <summary>The section table in file-table order.</summary>
    public IReadOnlyList<PackageSectionInfo> Sections { get; }

    /// <summary>The stored model: graphs, states, properties, metadata. Decoded on first access (the bytes are already in memory).</summary>
    /// <exception cref="InvalidDataException">The definition does not decode with this version of Sharplight (e.g. an unknown op).</exception>
    public DefinitionInfo Definition => _definition.Value;

    /// <summary>The weight index in name order.</summary>
    public IReadOnlyList<PackageWeightInfo> Weights { get; }

    /// <summary>Sum of the stored weight sizes.</summary>
    public long WeightBytes { get; }

    /// <summary>Stored artifact sets in section-table order (oldest first).</summary>
    public IReadOnlyList<ArtifactSetInfo> ArtifactSets { get; }

    /// <summary>Bytes <see cref="ModelPackage.Compact"/> would remove (superseded or stripped sets, old section tables).</summary>
    public long ReclaimableBytes { get; }
}
