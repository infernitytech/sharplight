using Sharplight.Models;

namespace Sharplight.Gguf;

/// <summary>The source of a tensor known only from a tensor table read before the file is available: it has a size and no data.</summary>
/// <param name="name">The tensor's name, for the error.</param>
/// <param name="length">The size of its data in bytes.</param>
internal sealed class GgufHeaderOnlySource(string name, long length) : IWeightSource
{
    /// <inheritdoc/>
    public long ByteLength => length;

    /// <inheritdoc/>
    public void Read(long offset, Span<byte> destination) => throw new GgufHeaderOnlyDataException(name);
}
