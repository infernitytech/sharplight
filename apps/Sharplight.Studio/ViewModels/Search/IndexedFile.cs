namespace Sharplight.Studio.ViewModels.Search;

/// <summary>A document of an index with the fingerprint that tells whether it changed.</summary>
/// <param name="Path">The path relative to the indexed folder, with <c>/</c> as the separator.</param>
/// <param name="Size">The size in bytes.</param>
/// <param name="ModifiedTicks">The last write time (UTC ticks).</param>
/// <param name="Hash">SHA-256 of the content in hex, or empty when the file was too large to read.</param>
/// <param name="Error">Why the file could not be indexed (too large, damaged), or empty.</param>
internal sealed record IndexedFile(string Path, long Size, long ModifiedTicks, string Hash, string Error = "");
