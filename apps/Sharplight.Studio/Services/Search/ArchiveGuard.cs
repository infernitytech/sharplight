using System.IO.Compression;

namespace Sharplight.Studio.Services.Search;

/// <summary>Refuses Office files (zip packages) that would unpack to too much, before any part is read.</summary>
internal static class ArchiveGuard
{
    /// <summary>Checks the package's part list against <paramref name="limits"/>.</summary>
    /// <param name="path">The package.</param>
    /// <param name="limits">The limits.</param>
    /// <exception cref="InvalidDataException">There are too many parts, they unpack to too much, or one compresses suspiciously well.</exception>
    public static void Check(string path, DocumentLimits limits)
    {
        using var archive = ZipFile.OpenRead(path);
        if (archive.Entries.Count > limits.MaxEntries)
        {
            throw new InvalidDataException($"The package has {archive.Entries.Count} parts (limit {limits.MaxEntries}).");
        }

        long total = 0;
        foreach (var entry in archive.Entries)
        {
            total += entry.Length;
            if (total > limits.MaxUnpackedBytes)
            {
                throw new InvalidDataException($"The package unpacks to more than {limits.MaxUnpackedBytes >> 20} MB.");
            }

            if (entry.Length > limits.RatioFloorBytes && entry.Length / Math.Max(1, entry.CompressedLength) > limits.MaxCompressionRatio)
            {
                throw new InvalidDataException($"The part '{entry.FullName}' is compressed {entry.Length / Math.Max(1, entry.CompressedLength)} to 1 (limit {limits.MaxCompressionRatio}).");
            }
        }
    }
}
