using System.Buffers;
using System.Buffers.Binary;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Gguf;

/// <summary>
/// An NVFP4 tensor with its per-tensor (or per-expert) output multiplier <c>X.scale</c>, read as <see cref="DataType.NVFP4_GS"/>: every
/// 36-byte ggml NVFP4 block { u8 d[4]; u8 qs[32] } gets the f32 multiplier of its matrix in front. llama.cpp multiplies the matmul
/// output by the multiplier (<c>build_lora_mm(w, x, w_s)</c>); carrying it in the block gives the same product up to f32 rounding
/// order, which folding it into the E4M3 block scales could not do without loss. The multiplier is read on the first read, so a
/// tensor table alone maps without data.
/// </summary>
internal sealed class GgufNvfp4ScaledSource : IWeightSource
{
    private const int SourceBlockBytes = 36;
    private const int BlockBytes = 40;

    private readonly IWeightSource _codes;
    private readonly WeightInfo _scale;
    private readonly long _blocksPerMatrix;
    private readonly Lazy<float[]> _multipliers;

    /// <param name="codes">The NVFP4 tensor bytes.</param>
    /// <param name="scale">The multiplier tensor: one value, or one per matrix of <paramref name="codes"/> (F32, F16 or BF16).</param>
    /// <param name="blocksPerMatrix">64-value blocks per matrix.</param>
    public GgufNvfp4ScaledSource(IWeightSource codes, WeightInfo scale, long blocksPerMatrix)
    {
        _codes = codes;
        _scale = scale;
        _blocksPerMatrix = blocksPerMatrix;
        ByteLength = codes.ByteLength / SourceBlockBytes * BlockBytes;
        _multipliers = new(ReadMultipliers);
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

        var multipliers = _multipliers.Value;
        long first = offset / BlockBytes;
        long last = (offset + destination.Length - 1) / BlockBytes;
        int count = checked((int)(last - first + 1));
        byte[] source = ArrayPool<byte>.Shared.Rent(count * SourceBlockBytes);
        try
        {
            _codes.Read(first * SourceBlockBytes, source.AsSpan(0, count * SourceBlockBytes));
            Span<byte> block = stackalloc byte[BlockBytes];
            int written = 0;
            for (int i = 0; i < count; i++)
            {
                long b = first + i;
                float g = multipliers.Length == 1 ? multipliers[0] : multipliers[b / _blocksPerMatrix];
                BinaryPrimitives.WriteSingleLittleEndian(block, g);
                source.AsSpan(i * SourceBlockBytes, SourceBlockBytes).CopyTo(block[4..]);
                long blockStart = b * BlockBytes;
                int from = (int)Math.Max(0, offset - blockStart);
                int to = (int)Math.Min(BlockBytes, offset + destination.Length - blockStart);
                block[from..to].CopyTo(destination[written..]);
                written += to - from;
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(source);
        }
    }

    private float[] ReadMultipliers()
    {
        var bytes = new byte[_scale.Source.ByteLength];
        _scale.Source.Read(0, bytes);
        int width = _scale.Type == DataType.F32 ? 4 : 2;
        var values = new float[bytes.Length / width];
        for (int i = 0; i < values.Length; i++)
        {
            var at = bytes.AsSpan(i * width);
            values[i] = _scale.Type == DataType.F32 ? BinaryPrimitives.ReadSingleLittleEndian(at)
                : _scale.Type == DataType.F16 ? (float)BinaryPrimitives.ReadHalfLittleEndian(at)
                : BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadUInt16LittleEndian(at) << 16);
        }

        return values;
    }
}
