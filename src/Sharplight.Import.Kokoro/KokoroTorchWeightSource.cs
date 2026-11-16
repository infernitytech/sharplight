using Sharplight.Import.PyTorch;
using Sharplight.Models;

namespace Sharplight.Import.Kokoro;

/// <summary>Zero-copy view of one tensor inside a memory-mapped PyTorch checkpoint; reads fail once the file is disposed.</summary>
internal sealed unsafe class KokoroTorchWeightSource(TorchCheckpointFile file, TorchTensorEntry entry) : IWeightSource
{
    /// <inheritdoc/>
    public long ByteLength => entry.ByteLength;

    /// <inheritdoc/>
    public void Read(long position, Span<byte> destination)
    {
        if (position < 0 || position + destination.Length > entry.ByteLength)
        {
            throw new ArgumentOutOfRangeException(nameof(position));
        }

        new ReadOnlySpan<byte>(file.BasePointer + entry.FileOffset + position, destination.Length).CopyTo(destination);
    }

    /// <inheritdoc/>
    public nint TryGetPointer() => (nint)(file.BasePointer + entry.FileOffset);
}
