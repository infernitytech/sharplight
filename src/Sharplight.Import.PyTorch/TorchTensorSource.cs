using Sharplight.Models;

namespace Sharplight.Import.PyTorch;

/// <summary>Zero-copy view of a tensor's bytes inside a memory-mapped PyTorch checkpoint; reads fail once the file is disposed.</summary>
/// <param name="file">The open checkpoint.</param>
/// <param name="fileOffset">Offset of the tensor's first byte in the file.</param>
/// <param name="length">Size of the tensor in bytes.</param>
internal sealed unsafe class TorchTensorSource(TorchCheckpointFile file, long fileOffset, long length) : IWeightSource
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
