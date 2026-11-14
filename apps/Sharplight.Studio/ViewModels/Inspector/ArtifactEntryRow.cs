using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Inspection;

namespace Sharplight.Studio.ViewModels.Inspector;

/// <summary>One entry of an artifact set.</summary>
/// <param name="Key">Entry key (e.g. <c>kernel/&lt;content hash&gt;</c>).</param>
/// <param name="Bytes">Size in bytes.</param>
/// <param name="Offset">Absolute file offset.</param>
internal sealed record ArtifactEntryRow(string Key, long Bytes, long Offset) : IFilterableRow
{
    private const int MinHashLength = 8;

    /// <summary>The content hash of a content-addressed key (its last segment when it is hexadecimal), else empty.</summary>
    public string Hash
    {
        get
        {
            var last = Key[(Key.LastIndexOf('/') + 1)..];
            return last.Length >= MinHashLength && last.All(char.IsAsciiHexDigit) ? last : "";
        }
    }

    /// <summary>The key without the hash, e.g. <c>kernel/</c>.</summary>
    public string Prefix => Hash.Length == 0 ? Key : Key[..^Hash.Length];

    /// <summary>The size for display.</summary>
    public string Size => ByteFormat.Format(Bytes);

    /// <summary>The offset for display.</summary>
    public string OffsetText => ByteFormat.Number(Offset);

    /// <inheritdoc/>
    public bool Matches(string[] terms) => TextFilter.Matches(terms, Key);
}
