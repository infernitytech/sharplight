using Sharplight.Models;

namespace Sharplight.Gguf;

/// <summary>Zero-copy view of a byte range inside a memory-mapped GGUF file; reads fail once the file is disposed.</summary>
internal sealed unsafe class GgufWeightSource(GgufFile file, long offset, long length) : IWeightSource
{
    /// <inheritdoc/>
    public long ByteLength => length;

    /// <inheritdoc/>
    public void Read(long off, Span<byte> destination)
    {
        if (off < 0 || off + destination.Length > length)
        {
            throw new ArgumentOutOfRangeException(nameof(off));
        }

        file.ThrowIfDisposed();
        new ReadOnlySpan<byte>(file.BasePointer + offset + off, destination.Length).CopyTo(destination);
    }

    /// <inheritdoc/>
    public nint TryGetPointer()
    {
        file.ThrowIfDisposed();
        return (nint)(file.BasePointer + offset);
    }
}
