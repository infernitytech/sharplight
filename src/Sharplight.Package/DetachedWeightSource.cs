using Sharplight.Models;

namespace Sharplight.Package;

/// <summary>Weight source of a definition decoded for inspection only: knows its size, refuses reads.</summary>
internal sealed class DetachedWeightSource(long byteLength) : IWeightSource
{
    public long ByteLength => byteLength;

    public void Read(long offset, Span<byte> destination)
        => throw new InvalidOperationException("Weights of an inspected package are not readable; open the package with ModelPackageImporter to load them.");
}
