using FFmpeg.AutoGen;

namespace Sharplight.Video.Ffmpeg;

/// <summary>
/// The output side shared by the streams of one encode: the muxer writing into a .NET stream, encoding a frame into packets and writing
/// packets interleaved by time. Errors of the destination stream surface as <see cref="IOException"/>. Not thread safe.
/// </summary>
internal sealed unsafe class FfmpegMuxer : IDisposable
{
    private readonly OutputStreamIo _io;
    private AVFormatContext* _format;
    private AVPacket* _packet;

    /// <param name="muxer">The FFmpeg muxer name.</param>
    /// <param name="output">The destination; not disposed.</param>
    public FfmpegMuxer(string muxer, Stream output)
    {
        _io = new OutputStreamIo(output);
        AVFormatContext* format = null;
        try
        {
            Check(ffmpeg.avformat_alloc_output_context2(&format, null, muxer, null), $"create the {muxer} muxer");
            _format = format;
            _format->pb = _io.Context;
            _packet = ffmpeg.av_packet_alloc();
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    /// <summary>The muxer.</summary>
    public AVFormatContext* Format => _format;

    /// <summary>Whether the muxer wants the codecs' global headers.</summary>
    public bool NeedsGlobalHeader => (_format->oformat->flags & ffmpeg.AVFMT_GLOBALHEADER) != 0;

    /// <summary>Whether the muxer stores packets of a codec as they are (normal compliance).</summary>
    public bool Accepts(AVCodecID codec) => ffmpeg.avformat_query_codec(_format->oformat, codec, ffmpeg.FF_COMPLIANCE_NORMAL) == 1;

    /// <summary>Encodes <paramref name="frame"/> (or drains the encoder for <see langword="null"/>) and writes the packets.</summary>
    public void Send(AVCodecContext* encoder, AVStream* stream, AVFrame* frame)
    {
        Check(ffmpeg.avcodec_send_frame(encoder, frame), "encode a frame");
        while (true)
        {
            int result = ffmpeg.avcodec_receive_packet(encoder, _packet);
            if (FfmpegError.IsAgainOrEnd(result))
            {
                return;
            }

            Check(result, "encode a frame");
            ffmpeg.av_packet_rescale_ts(_packet, encoder->time_base, stream->time_base);
            _packet->stream_index = stream->index;
            Write(_packet);
        }
    }

    /// <summary>Writes a packet whose stream index and timestamps are set (in its stream's time base); the packet is unreferenced.</summary>
    public void Write(AVPacket* packet) => Check(ffmpeg.av_interleaved_write_frame(_format, packet), "write a packet");

    /// <summary>Throws for a negative result: the destination's own exception when a write failed, else FFmpeg's message.</summary>
    /// <returns>The non-negative result.</returns>
    public int Check(int result, string action)
    {
        if (result < 0)
        {
            ThrowIfOutputFailed();
        }

        return FfmpegError.Check(result, action);
    }

    /// <summary>Rethrows the destination's exception, if a write failed.</summary>
    public void ThrowIfOutputFailed()
    {
        if (_io.Failure is not null)
        {
            throw new IOException($"Writing the video failed: {_io.Failure.Message}", _io.Failure);
        }
    }

    /// <summary>Writes the trailer and flushes the destination.</summary>
    public void Finish()
    {
        Check(ffmpeg.av_write_trailer(_format), "finish the file");
        ffmpeg.avio_flush(_io.Context);
        ThrowIfOutputFailed();
    }

    /// <inheritdoc />
    public void Dispose()
    {
        AVPacket* packet = _packet;
        ffmpeg.av_packet_free(&packet);
        _packet = null;
        if (_format is not null)
        {
            _format->pb = null;
            ffmpeg.avformat_free_context(_format);
            _format = null;
        }

        _io.Dispose();
    }
}
