using System.Buffers;
using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// A weight whose rows are re-encoded from a checkpoint's packed storage while they are read (packed 4-bit codes and their scale
/// tensors into a block-quantized type). Each read converts only the rows it touches from the memory-mapped source tensors and
/// nothing is cached, so a large checkpoint does not stay resident in host memory after upload. Thread-safe when the converter is.
/// </summary>
internal sealed class HfRowConvertedSource : IWeightSource
{
    private readonly long _rows;
    private readonly int _rowBytes;
    private readonly Action<long, Span<byte>> _convertRow;

    /// <param name="rows">Number of rows (all leading dimensions flattened).</param>
    /// <param name="rowBytes">Bytes of one converted row.</param>
    /// <param name="convertRow">Writes converted row <c>r</c> (the first argument) into the span, which is exactly one row long.</param>
    public HfRowConvertedSource(long rows, long rowBytes, Action<long, Span<byte>> convertRow)
    {
        _rows = rows;
        _rowBytes = checked((int)rowBytes);
        _convertRow = convertRow;
    }

    /// <inheritdoc/>
    public long ByteLength => _rows * _rowBytes;

    /// <inheritdoc/>
    public void Read(long offset, Span<byte> destination)
    {
        if (offset < 0 || offset + destination.Length > ByteLength)
        {
            throw new ArgumentOutOfRangeException(nameof(offset));
        }

        byte[]? partial = null;
        try
        {
            long position = offset;
            int written = 0;
            while (written < destination.Length)
            {
                long row = position / _rowBytes;
                int inRow = (int)(position - row * _rowBytes);
                int take = Math.Min(_rowBytes - inRow, destination.Length - written);
                if (inRow == 0 && take == _rowBytes)
                {
                    _convertRow(row, destination.Slice(written, _rowBytes));
                }
                else
                {
                    partial ??= ArrayPool<byte>.Shared.Rent(_rowBytes);
                    _convertRow(row, partial.AsSpan(0, _rowBytes));
                    partial.AsSpan(inRow, take).CopyTo(destination[written..]);
                }

                written += take;
                position += take;
            }
        }
        finally
        {
            if (partial is not null)
            {
                ArrayPool<byte>.Shared.Return(partial);
            }
        }
    }
}
