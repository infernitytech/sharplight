namespace Sharplight.ModelFeed;

/// <summary>One upstream repository at a pinned revision.</summary>
public sealed record ProvenanceSource
{
    /// <summary>Repository id (<c>Qwen/Qwen3-0.6B-GGUF</c>).</summary>
    public required string Repository { get; init; }

    /// <summary>Pinned commit.</summary>
    public required string Revision { get; init; }

    /// <summary>Files used, with checksums.</summary>
    public IReadOnlyList<ProvenanceFile> Files { get; init; } = [];

    /// <summary><c>primary</c> or the companion's auxiliary role.</summary>
    public string? Role { get; init; }
}
