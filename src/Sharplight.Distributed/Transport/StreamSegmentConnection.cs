using System.Buffers;

namespace Sharplight.Distributed.Transport;

/// <summary>A frame connection over a byte stream (a TCP socket's network stream, or a TLS stream over it).</summary>
internal sealed class StreamSegmentConnection : ISegmentConnection
{
    /// <summary>Payloads up to this size are sent with their header in one write.</summary>
    private const int CoalesceBytes = 256 << 10;

    private readonly Stream _stream;
    private readonly IDisposable? _owner;
    private readonly int _maxFrameBytes;
    private readonly Lock _sendLock = new();
    private readonly byte[] _receiveHeader = new byte[FrameHeader.Size];
    private int _disposed;

    /// <param name="stream">The stream; the connection owns it.</param>
    /// <param name="owner">Disposed with the stream (the socket), or null.</param>
    /// <param name="remoteName">The other end, for messages.</param>
    /// <param name="maxFrameBytes">Largest payload accepted.</param>
    public StreamSegmentConnection(Stream stream, IDisposable? owner, string remoteName, int maxFrameBytes)
    {
        _stream = stream;
        _owner = owner;
        _maxFrameBytes = maxFrameBytes;
        RemoteName = remoteName;
    }

    public string RemoteName { get; }

    public void Send(FrameKind kind, uint requestId, ReadOnlySpan<byte> payload)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        lock (_sendLock)
        {
            try
            {
                if (payload.Length <= CoalesceBytes)
                {
                    int total = FrameHeader.Size + payload.Length;
                    var buffer = ArrayPool<byte>.Shared.Rent(total);
                    try
                    {
                        FrameHeader.Write(buffer, kind, requestId, payload.Length);
                        payload.CopyTo(buffer.AsSpan(FrameHeader.Size));
                        _stream.Write(buffer, 0, total);
                    }
                    finally
                    {
                        ArrayPool<byte>.Shared.Return(buffer);
                    }
                }
                else
                {
                    Span<byte> header = stackalloc byte[FrameHeader.Size];
                    FrameHeader.Write(header, kind, requestId, payload.Length);
                    _stream.Write(header);
                    _stream.Write(payload);
                }

                _stream.Flush();
            }
            catch (Exception e) when (e is not IOException)
            {
                throw new IOException($"Sending to {RemoteName} failed: {e.Message}", e);
            }
        }
    }

    public Frame Receive()
    {
        try
        {
            _stream.ReadExactly(_receiveHeader);
            var (kind, requestId, length) = FrameHeader.Read(_receiveHeader, _maxFrameBytes);
            // Not zeroed: every byte is read into it.
            var payload = length == 0 ? [] : GC.AllocateUninitializedArray<byte>(length);
            _stream.ReadExactly(payload);
            return new Frame(kind, requestId, payload);
        }
        catch (EndOfStreamException e)
        {
            throw new IOException($"{RemoteName} closed the connection.", e);
        }
        catch (Exception e) when (e is InvalidDataException or ObjectDisposedException or System.Net.Sockets.SocketException)
        {
            throw new IOException($"The connection to {RemoteName} broke: {e.Message}", e);
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        _stream.Dispose();
        _owner?.Dispose();
    }
}
