namespace Sharplight.Hub.Tests;

/// <summary>Reads from an inner stream and throws an <see cref="IOException"/> (a dropped connection) after a number of bytes.</summary>
internal sealed class FailingStream(Stream inner, long failAfter) : Stream
{
    private long _position;

    public override bool CanRead => true;

    public override bool CanSeek => false;

    public override bool CanWrite => false;

    public override long Length => inner.Length;

    public override long Position
    {
        get => _position;
        set => throw new NotSupportedException();
    }

    public override int Read(byte[] buffer, int offset, int count)
    {
        if (_position >= failAfter)
        {
            throw new IOException("connection reset (test)");
        }

        int read = inner.Read(buffer, offset, (int)Math.Min(count, failAfter - _position));
        _position += read;
        return read;
    }

    public override void Flush()
    {
    }

    public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();

    public override void SetLength(long value) => throw new NotSupportedException();

    public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            inner.Dispose();
        }

        base.Dispose(disposing);
    }
}
