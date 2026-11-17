namespace Sharplight.ModelFeed.Tests;

/// <summary>Returns some bytes, then fails like a dropped connection.</summary>
internal sealed class BreakingStream(ReadOnlyMemory<byte> data) : Stream
{
    private int _position;

    public override bool CanRead => true;

    public override bool CanSeek => false;

    public override bool CanWrite => false;

    public override long Length => throw new NotSupportedException();

    public override long Position
    {
        get => throw new NotSupportedException();
        set => throw new NotSupportedException();
    }

    public override int Read(byte[] buffer, int offset, int count)
    {
        if (_position >= data.Length)
        {
            throw new IOException("Connection dropped.");
        }

        var length = Math.Min(count, data.Length - _position);
        data.Span.Slice(_position, length).CopyTo(buffer.AsSpan(offset, length));
        _position += length;
        return length;
    }

    public override void Flush()
    {
    }

    public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();

    public override void SetLength(long value) => throw new NotSupportedException();

    public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
}
