using System.Buffers;
using System.Buffers.Binary;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.HuggingFace;

/// <summary>
/// An FP8 checkpoint weight (E4M3 codes [batch..., N, K] plus separate scales) converted on the fly while it is read: to
/// <see cref="DataType.F8E4M3_B128"/> / <see cref="DataType.F8E4M3_B32"/> blocks (the scale written in front of every block of
/// codes, so the bytes are a pure re-layout), or to <see cref="DataType.BF16"/> values when the scale granularity does not fit a
/// block. Nothing is cached: every read converts the requested range from the memory-mapped codes, so a model of tens of GB does
/// not stay resident in host memory after upload. Thread-safe.
/// </summary>
internal sealed unsafe class HfFp8WeightSource : IWeightSource
{
    private readonly IWeightSource _codes;
    private readonly Lazy<float[]> _scales;
    private readonly HfFp8ScaleLayout _layout;
    private readonly DataType _target;
    private readonly long _rows;
    private readonly long _columns;

    /// <param name="codes">The E4M3 codes, row-major [batch..., N, K].</param>
    /// <param name="scales">The scale tensor (any float type), read once on first use.</param>
    /// <param name="layout">How the scales cover the values.</param>
    /// <param name="target">The produced type: <see cref="DataType.F8E4M3_B128"/>, <see cref="DataType.F8E4M3_B32"/> or <see cref="DataType.BF16"/>.</param>
    /// <param name="rows">Rows N of each matrix.</param>
    /// <param name="columns">Columns K (must be a multiple of the target block size).</param>
    public HfFp8WeightSource(IWeightSource codes, WeightInfo scales, HfFp8ScaleLayout layout, DataType target, long rows, long columns)
    {
        if (columns % target.BlockSize != 0)
        {
            throw new ArgumentException($"{columns} columns are not a multiple of the {target} block size.");
        }

        _codes = codes;
        _scales = new(() => HfTransforms.DecodeF32(scales));
        _layout = layout;
        _target = target;
        _rows = rows;
        _columns = columns;
        ByteLength = target.ByteSize(codes.ByteLength);
    }

    /// <inheritdoc/>
    public long ByteLength { get; }

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

        long unitBytes = _target.BlockBytes;
        long firstUnit = offset / unitBytes;
        long endUnit = (offset + destination.Length + unitBytes - 1) / unitBytes;
        bool aligned = offset % unitBytes == 0 && destination.Length % unitBytes == 0;
        int producedBytes = checked((int)((endUnit - firstUnit) * unitBytes));
        int codeCount = checked((int)((endUnit - firstUnit) * _target.BlockSize));
        byte[]? temporary = aligned ? null : ArrayPool<byte>.Shared.Rent(producedBytes);
        byte[]? codeBuffer = null;
        try
        {
            ReadOnlySpan<byte> codes;
            nint pointer = _codes.TryGetPointer();
            if (pointer != 0)
            {
                codes = new ReadOnlySpan<byte>((byte*)pointer + firstUnit * _target.BlockSize, codeCount);
            }
            else
            {
                codeBuffer = ArrayPool<byte>.Shared.Rent(codeCount);
                _codes.Read(firstUnit * _target.BlockSize, codeBuffer.AsSpan(0, codeCount));
                codes = codeBuffer.AsSpan(0, codeCount);
            }

            var output = aligned ? destination : temporary.AsSpan(0, producedBytes);
            Convert(firstUnit, codes, output);
            if (!aligned)
            {
                output.Slice((int)(offset - firstUnit * unitBytes), destination.Length).CopyTo(destination);
            }
        }
        finally
        {
            if (temporary is not null)
            {
                ArrayPool<byte>.Shared.Return(temporary);
            }

            if (codeBuffer is not null)
            {
                ArrayPool<byte>.Shared.Return(codeBuffer);
            }
        }
    }

    /// <summary>Writes the output units (blocks, or BF16 values) from <paramref name="firstUnit"/> on, one per <see cref="DataType.BlockSize"/> codes.</summary>
    private void Convert(long firstUnit, ReadOnlySpan<byte> codes, Span<byte> output)
    {
        var scales = _scales.Value;
        int block = _target.BlockSize;
        int unitBytes = _target.BlockBytes;
        long unitsPerRow = _columns / block;
        int units = codes.Length / block;
        for (int u = 0; u < units; u++)
        {
            long unit = firstUnit + u;
            long globalRow = unit / unitsPerRow;
            long column = unit % unitsPerRow * block;
            float scale = _layout.At(scales, globalRow / _rows, globalRow % _rows, column);
            var unitOutput = output.Slice(u * unitBytes, unitBytes);
            if (_target == DataType.BF16)
            {
                BinaryPrimitives.WriteUInt16LittleEndian(unitOutput, HfFp8Codec.ToBFloat16(HfFp8Codec.E4m3(codes[u]) * scale));
            }
            else
            {
                BinaryPrimitives.WriteSingleLittleEndian(unitOutput, scale);
                codes.Slice(u * block, block).CopyTo(unitOutput[4..]);
            }
        }
    }
}
