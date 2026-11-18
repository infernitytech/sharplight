namespace Sharplight.Studio.Services.Search;

/// <summary>What a document may cost to read: files above these limits are skipped and recorded as such in the index.</summary>
internal sealed record DocumentLimits
{
    /// <summary>The largest plain text, Markdown, HTML or source file read.</summary>
    public long MaxTextBytes { get; init; } = 16L << 20;

    /// <summary>The largest PDF or Office file read (compressed, as on disk).</summary>
    public long MaxPackageBytes { get; init; } = 128L << 20;

    /// <summary>The most an Office file may unpack to, summed over its parts (zip bomb guard).</summary>
    public long MaxUnpackedBytes { get; init; } = 512L << 20;

    /// <summary>The most parts an Office file may have.</summary>
    public int MaxEntries { get; init; } = 10_000;

    /// <summary>The compression ratio above which a part of more than <see cref="RatioFloorBytes"/> bytes is refused (zip bomb guard).</summary>
    public int MaxCompressionRatio { get; init; } = 200;

    /// <summary>Parts up to this many unpacked bytes are not checked against <see cref="MaxCompressionRatio"/> (small XML compresses well).</summary>
    public long RatioFloorBytes { get; init; } = 8L << 20;

    /// <summary>The most PDF pages, slides or sheets read from one file.</summary>
    public int MaxPages { get; init; } = 3000;

    /// <summary>The most characters kept from one file.</summary>
    public int MaxCharacters { get; init; } = 20_000_000;

    /// <summary>The most cells read from one sheet.</summary>
    public int MaxCells { get; init; } = 500_000;
}
