using Sharplight.Models;

namespace Sharplight.Package.Tests;

/// <summary>An in-memory byte source (for attachments and weights built in tests).</summary>
public sealed class BytesSource(byte[] bytes) : IWeightSource
{
    public long ByteLength => bytes.Length;

    public void Read(long offset, Span<byte> destination) => bytes.AsSpan(checked((int)offset), destination.Length).CopyTo(destination);
}
