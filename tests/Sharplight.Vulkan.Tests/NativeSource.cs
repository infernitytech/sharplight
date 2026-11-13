using System.Runtime.InteropServices;
using Sharplight.Models;

namespace Sharplight.Vulkan.Tests;

/// <summary>A weight source in native memory, for weights larger than a managed array can hold.</summary>
internal sealed unsafe class NativeSource : IWeightSource, IDisposable
{
    private byte* _data;

    /// <summary>Allocates <paramref name="bytes"/> bytes and lets <paramref name="fill"/> write them.</summary>
    public NativeSource(long bytes, Action<nint, long> fill)
    {
        ByteLength = bytes;
        _data = (byte*)NativeMemory.Alloc((nuint)bytes);
        fill((nint)_data, bytes);
    }

    public long ByteLength { get; }

    public void Read(long offset, Span<byte> destination) => new ReadOnlySpan<byte>(_data + offset, destination.Length).CopyTo(destination);

    public nint TryGetPointer() => (nint)_data;

    public void Dispose()
    {
        NativeMemory.Free(_data);
        _data = null;
    }
}
