using Sharplight.Models;

namespace Sharplight.Import.WhisperCpp;

/// <summary>Zero-copy view of a byte range inside a memory-mapped whisper.cpp file; reads fail once the file is disposed.</summary>
internal sealed unsafe class WhisperCppWeightSource(WhisperCppFile file, long offset, long length) : IWeightSource
{
    /// <inheritdoc/>
    public long ByteLength => length;

    /// <inheritdoc/>
    public void Read(long position, Span<byte> destination)
    {
        if (position < 0 || position + destination.Length > length)
        {
            throw new ArgumentOutOfRangeException(nameof(position));
        }

        new ReadOnlySpan<byte>(file.Pointer(offset + position), destination.Length).CopyTo(destination);
    }

    /// <inheritdoc/>
    public nint TryGetPointer() => (nint)file.Pointer(offset);
}
