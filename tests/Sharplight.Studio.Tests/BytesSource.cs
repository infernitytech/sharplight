using Sharplight.Models;

namespace Sharplight.Studio.Tests;

/// <summary>An in-memory byte source for package attachments built in tests.</summary>
internal sealed class BytesSource(byte[] bytes) : IWeightSource
{
    public long ByteLength => bytes.Length;

    public void Read(long offset, Span<byte> destination) => bytes.AsSpan(checked((int)offset), destination.Length).CopyTo(destination);
}
