using System.Buffers;
using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// A weight whose rows are split into equal column groups that are reordered: output group g of every row is source group
/// <c>order[g]</c>. Computed while read (nothing is cached), so it adds no resident host memory. Groups must be whole storage
/// blocks. Thread-safe.
/// </summary>
/// <param name="source">The stored weight.</param>
/// <param name="rowBytes">Bytes per row.</param>
/// <param name="groupBytes">Bytes per column group; <paramref name="order"/>.Length groups make a row.</param>
/// <param name="order">For each output group, the source group it takes.</param>
internal sealed class HfColumnGroupPermutationSource(IWeightSource source, long rowBytes, long groupBytes, int[] order) : IWeightSource
{
    /// <inheritdoc/>
    public long ByteLength => source.ByteLength;

    /// <inheritdoc/>
    public void Materialize() => source.Materialize();

    /// <inheritdoc/>
    public void Read(long offset, Span<byte> destination)
    {
        if (offset < 0 || offset + destination.Length > ByteLength)
        {
            throw new ArgumentOutOfRangeException(nameof(offset));
        }

        if (destination.IsEmpty)
        {
            return;
        }

        long firstRow = offset / rowBytes;
        long endRow = (offset + destination.Length + rowBytes - 1) / rowBytes;
        int bytes = checked((int)((endRow - firstRow) * rowBytes));
        var rows = ArrayPool<byte>.Shared.Rent(bytes);
        var permuted = ArrayPool<byte>.Shared.Rent(bytes);
        try
        {
            source.Read(firstRow * rowBytes, rows.AsSpan(0, bytes));
            for (long row = 0; row < endRow - firstRow; row++)
            {
                for (int group = 0; group < order.Length; group++)
                {
                    rows.AsSpan(checked((int)(row * rowBytes + order[group] * groupBytes)), (int)groupBytes)
                        .CopyTo(permuted.AsSpan(checked((int)(row * rowBytes + group * groupBytes))));
                }
            }

            permuted.AsSpan(checked((int)(offset - firstRow * rowBytes)), destination.Length).CopyTo(destination);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(rows);
            ArrayPool<byte>.Shared.Return(permuted);
        }
    }
}
