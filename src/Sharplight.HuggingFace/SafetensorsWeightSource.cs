using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>Zero-copy view of a byte range inside a memory-mapped safetensors file; reads fail once the file is disposed.</summary>
internal sealed unsafe class SafetensorsWeightSource(SafetensorsFile file, long fileOffset, long length) : IWeightSource
{
    /// <inheritdoc/>
    public long ByteLength => length;

    /// <inheritdoc/>
    public void Read(long offset, Span<byte> destination)
    {
        if (offset < 0 || offset + destination.Length > length)
        {
            throw new ArgumentOutOfRangeException(nameof(offset));
        }

        new ReadOnlySpan<byte>(file.BasePointer + fileOffset + offset, destination.Length).CopyTo(destination);
    }

    /// <inheritdoc/>
    public nint TryGetPointer() => (nint)(file.BasePointer + fileOffset);
}
