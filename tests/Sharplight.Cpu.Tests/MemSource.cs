using System.Runtime.InteropServices;
using Sharplight.Models;

namespace Sharplight.Cpu.Tests;

/// <summary>Weight bytes in managed memory; optionally exposes a pinned pointer to exercise the zero-copy path.</summary>
internal sealed class MemSource : IWeightSource, IDisposable
{
    private readonly byte[] _data;
    private readonly GCHandle _h;
    private readonly bool _zeroCopy;
    public MemSource(byte[] data, bool zeroCopy = true) { _data = data; _zeroCopy = zeroCopy; if (zeroCopy)
        {
            _h = GCHandle.Alloc(data, GCHandleType.Pinned);
        }
    }
    public long ByteLength => _data.Length;
    public void Read(long offset, Span<byte> destination) => _data.AsSpan((int)offset, destination.Length).CopyTo(destination);
    public nint TryGetPointer() => _zeroCopy ? _h.AddrOfPinnedObject() : 0;
    public void Dispose() { if (_zeroCopy && _h.IsAllocated)
        {
            _h.Free();
        }
    }
}
