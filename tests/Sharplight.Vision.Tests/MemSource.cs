using System.Runtime.InteropServices;
using Sharplight.Models;

namespace Sharplight.Vision.Tests;

/// <summary>Weight bytes in managed memory.</summary>
internal sealed class MemSource : IWeightSource
{
    private readonly byte[] _data; private readonly GCHandle _h;
    public MemSource(byte[] data) { _data = data; _h = GCHandle.Alloc(data, GCHandleType.Pinned); }
    public long ByteLength => _data.Length;
    public void Read(long offset, Span<byte> destination) => _data.AsSpan((int)offset, destination.Length).CopyTo(destination);
    public nint TryGetPointer() => _h.AddrOfPinnedObject();
}
