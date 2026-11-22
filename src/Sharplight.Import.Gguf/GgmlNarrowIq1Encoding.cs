using Sharplight.Models;

namespace Sharplight.Import.Gguf;

/// <summary>The stored blocks of a narrow IQ1 type (ggml ids 64-66) as an <see cref="IStoredBlockEncoding"/> of IQ1_S blocks (<see cref="GgmlNarrowIq1.Widen"/>).</summary>
internal sealed class GgmlNarrowIq1Encoding : IStoredBlockEncoding
{
    private readonly int _ggmlType;

    /// <param name="ggmlType">The narrow type id.</param>
    public GgmlNarrowIq1Encoding(int ggmlType)
    {
        _ggmlType = ggmlType;
        StoredBlockBytes = GgmlNarrowIq1.StoredBlockBytes(ggmlType);
    }

    /// <inheritdoc/>
    public int StoredBlockBytes { get; }

    /// <inheritdoc/>
    public int BlockBytes => GgmlNarrowIq1.WideBlockBytes;

    /// <inheritdoc/>
    public void Decode(ReadOnlySpan<byte> stored, Span<byte> blocks)
    {
        int count = stored.Length / StoredBlockBytes;
        if (count * StoredBlockBytes != stored.Length || blocks.Length != count * BlockBytes)
        {
            throw new ArgumentException($"{stored.Length} stored bytes do not decode to {blocks.Length} bytes of IQ1_S blocks.");
        }

        for (int b = 0; b < count; b++)
        {
            GgmlNarrowIq1.Widen(_ggmlType, stored.Slice(b * StoredBlockBytes, StoredBlockBytes), blocks.Slice(b * BlockBytes, BlockBytes));
        }
    }
}
