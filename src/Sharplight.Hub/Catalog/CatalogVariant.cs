namespace Sharplight.Hub.Catalog;

/// <summary>One downloadable form of a catalog entry (a format and quantization) with its companions.</summary>
public sealed record CatalogVariant : CatalogFileSet
{
    /// <summary>Unique within the entry, e.g. <c>q4-k-m-gguf</c>.</summary>
    public required string Id { get; init; }

    /// <summary>What a user picks, e.g. <c>Q4_K_M GGUF</c>.</summary>
    public required string Label { get; init; }

    /// <summary>The file format, one of <see cref="CatalogFormats.All"/>.</summary>
    public required string Format { get; init; }

    /// <summary>The weight quantization (<c>Q4_K_M</c>, <c>BF16</c>, <c>FP8</c>, ...), or <see langword="null"/> when not meaningful.</summary>
    public string? Quantization { get; init; }

    /// <summary>The variant offered first.</summary>
    public bool Recommended { get; init; }

    /// <summary>Approximate device memory needed to run it with a short context, in bytes (0 when unknown).</summary>
    public long MemoryBytes { get; init; }

    /// <summary>Ids of the entry's <see cref="CatalogEntry.Companions"/> installed with this variant.</summary>
    public IReadOnlyList<string> Companions { get; init; } = [];

    /// <summary>
    /// The verification level of this variant when it is lower than the entry's (<see cref="CatalogLevel.Partial"/> for a variant whose files
    /// were not compared with the reference themselves), or <see langword="null"/> for the entry's level. See <see cref="CatalogEntry.LevelOf"/>.
    /// </summary>
    public CatalogLevel? Level { get; init; }

    /// <summary>What was verified on this variant's own files (or why not), or <see langword="null"/> when the entry's verification covers it.</summary>
    public string? Verification { get; init; }

    /// <summary>Quality relative to the entry's <see cref="CatalogEntry.ReferencePrecision"/>, one of <see cref="CatalogQualityTiers.All"/>, or <see langword="null"/>.</summary>
    public string? Quality { get; init; }

    /// <summary>One sentence on the quality relative to the reference precision, or <see langword="null"/>.</summary>
    public string? QualityNote { get; init; }

    /// <summary>A limitation of this variant only, or <see langword="null"/>.</summary>
    public string? Notes { get; init; }
}
