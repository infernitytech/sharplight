using Sharplight.Models;

namespace Sharplight.Onnx;

/// <summary>A byte range of a <see cref="MappedFile"/> exposed as a zero-copy weight source.</summary>
internal sealed unsafe class MappedRegionSource(MappedFile file, long offset, long length) : IWeightSource
{
    public long ByteLength => length;
    public void Read(long off, Span<byte> destination)
    {
        if (off < 0 || off + destination.Length > length)
        {
            throw new ArgumentOutOfRangeException(nameof(off));
        }

        new ReadOnlySpan<byte>(file.Pointer + offset + off, destination.Length).CopyTo(destination);
    }
    public nint TryGetPointer() => (nint)(file.Pointer + offset);
}
