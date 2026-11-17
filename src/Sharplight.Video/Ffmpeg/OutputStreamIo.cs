using System.Runtime.InteropServices;
using FFmpeg.AutoGen;

namespace Sharplight.Video.Ffmpeg;

/// <summary>
/// An <c>AVIOContext</c> that writes to a .NET <see cref="Stream"/> (seeking it when it can seek, which lets the muxers patch their
/// headers; otherwise the caller picks a streamable muxer mode). The callbacks are unmanaged function pointers with the stream behind a
/// <see cref="GCHandle"/>, so nothing can be collected while FFmpeg holds them.
/// </summary>
internal sealed unsafe class OutputStreamIo : IDisposable
{
    private const int BufferSize = 1 << 16;
    private const int SeekSize = 0x10000;
    private const int IoError = -5;
    private readonly Stream _stream;
    private GCHandle _handle;
    private AVIOContext* _context;

    /// <param name="stream">The destination; not disposed.</param>
    public OutputStreamIo(Stream stream)
    {
        _stream = stream;
        _handle = GCHandle.Alloc(this);
        byte* buffer = (byte*)ffmpeg.av_malloc(BufferSize);
        var write = new avio_alloc_context_write_packet_func { Pointer = (nint)(delegate* unmanaged<void*, byte*, int, int>)&Write };
        var seek = new avio_alloc_context_seek_func { Pointer = stream.CanSeek ? (nint)(delegate* unmanaged<void*, long, int, long>)&Seek : 0 };
        _context = ffmpeg.avio_alloc_context(buffer, BufferSize, 1, (void*)GCHandle.ToIntPtr(_handle), default, write, seek);
        if (_context is null)
        {
            ffmpeg.av_free(buffer);
            _handle.Free();
            throw new OutOfMemoryException("FFmpeg could not allocate the output context.");
        }

        _context->seekable = stream.CanSeek ? ffmpeg.AVIO_SEEKABLE_NORMAL : 0;
    }

    /// <summary>The context to assign to <c>AVFormatContext.pb</c>.</summary>
    public AVIOContext* Context => _context;

    /// <summary>Whether the destination can seek.</summary>
    public bool CanSeek => _stream.CanSeek;

    /// <summary>The first exception a callback hit (rethrown by the writer after FFmpeg reports the failed write).</summary>
    public Exception? Failure { get; private set; }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_context is not null)
        {
            ffmpeg.avio_flush(_context);
            ffmpeg.av_freep(&_context->buffer);
            AVIOContext* context = _context;
            ffmpeg.avio_context_free(&context);
            _context = null;
        }

        if (_handle.IsAllocated)
        {
            _handle.Free();
        }
    }

    [UnmanagedCallersOnly]
    private static int Write(void* opaque, byte* buffer, int size)
    {
        var self = (OutputStreamIo)GCHandle.FromIntPtr((nint)opaque).Target!;
        try
        {
            self._stream.Write(new ReadOnlySpan<byte>(buffer, size));
            return size;
        }
        catch (Exception exception)
        {
            self.Failure ??= exception;
            return IoError;
        }
    }

    [UnmanagedCallersOnly]
    private static long Seek(void* opaque, long offset, int whence)
    {
        var self = (OutputStreamIo)GCHandle.FromIntPtr((nint)opaque).Target!;
        try
        {
            if ((whence & SeekSize) != 0)
            {
                return self._stream.Length;
            }

            SeekOrigin origin = (whence & 3) switch
            {
                1 => SeekOrigin.Current,
                2 => SeekOrigin.End,
                _ => SeekOrigin.Begin,
            };
            return self._stream.Seek(offset, origin);
        }
        catch (Exception exception)
        {
            self.Failure ??= exception;
            return IoError;
        }
    }
}
