namespace Sharplight.ModelFeed;

/// <summary>A file of a package version.</summary>
public sealed record PackageFile
{
    /// <summary>File name in the version folder.</summary>
    public required string Name { get; init; }

    /// <summary><see cref="PackageFileRoles.Package"/> for the <c>.slm</c>.</summary>
    public string? Role { get; init; }

    /// <summary>Size in bytes.</summary>
    public long Size { get; init; }

    /// <summary>SHA-256 (lower-case hex).</summary>
    public required string Sha256 { get; init; }

    /// <summary>Chunk size of <see cref="Chunks"/>.</summary>
    public long? ChunkSize { get; init; }

    /// <summary>SHA-256 of each chunk, for resumable, verified, multi-source transfer.</summary>
    public IReadOnlyList<string> Chunks { get; init; } = [];
}
