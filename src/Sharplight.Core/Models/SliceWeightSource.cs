namespace Sharplight.Models;

internal sealed class SliceWeightSource(IWeightSource parent, long offset, long length) : IWeightSource
{
    public long ByteLength => length;
    public void Read(long off, Span<byte> destination) => parent.Read(offset + off, destination);
    public nint TryGetPointer() => parent.TryGetPointer() is var p && p != 0 ? p + (nint)offset : 0;
}
