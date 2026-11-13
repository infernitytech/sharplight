using Sharplight.Models;

namespace Sharplight.Onnx;

/// <summary>An in-memory weight source (typed tensor data, tests).</summary>
internal sealed class ArrayWeightSource(byte[] data, int offset = 0, int length = -1) : IWeightSource
{
    private readonly int _length = length < 0 ? data.Length - offset : length;
    public long ByteLength => _length;
    public void Read(long off, Span<byte> destination)
    {
        if (off < 0 || off + destination.Length > _length)
        {
            throw new ArgumentOutOfRangeException(nameof(off));
        }

        data.AsSpan(offset + (int)off, destination.Length).CopyTo(destination);
    }
}
