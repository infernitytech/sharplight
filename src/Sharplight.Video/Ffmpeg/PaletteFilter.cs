using System.Globalization;
using FFmpeg.AutoGen;

namespace Sharplight.Video.Ffmpeg;

/// <summary>
/// The GIF filter graph: <c>split → palettegen (one palette for the clip) → paletteuse</c>, fed with RGB frames. palettegen needs every
/// frame before it can emit the palette, so the indexed frames only come out of <see cref="Finish"/>.
/// </summary>
internal sealed unsafe class PaletteFilter : IDisposable
{
    private readonly int _width;
    private readonly int _height;
    private AVFilterGraph* _graph;
    private AVFilterContext* _source;
    private AVFilterContext* _sink;
    private AVFrame* _input;
    private AVFrame* _output;

    /// <param name="width">Frame width.</param>
    /// <param name="height">Frame height.</param>
    /// <param name="timeBase">The time base of the frame timestamps.</param>
    /// <param name="dither">The paletteuse dithering (<c>none</c>, <c>sierra2_4a</c>, ...).</param>
    public PaletteFilter(int width, int height, AVRational timeBase, string dither)
    {
        _width = width;
        _height = height;
        _graph = ffmpeg.avfilter_graph_alloc();
        _input = ffmpeg.av_frame_alloc();
        _output = ffmpeg.av_frame_alloc();
        string arguments = string.Create(CultureInfo.InvariantCulture, $"video_size={width}x{height}:pix_fmt={(int)AVPixelFormat.AV_PIX_FMT_RGB24}:time_base={timeBase.num}/{timeBase.den}:pixel_aspect=1/1");
        AVFilterContext* source = null;
        AVFilterContext* sink = null;
        FfmpegError.Check(ffmpeg.avfilter_graph_create_filter(&source, ffmpeg.avfilter_get_by_name("buffer"), "in", arguments, null, _graph), "create the filter input");
        FfmpegError.Check(ffmpeg.avfilter_graph_create_filter(&sink, ffmpeg.avfilter_get_by_name("buffersink"), "out", null, null, _graph), "create the filter output");
        _source = source;
        _sink = sink;

        AVFilterInOut* outputs = ffmpeg.avfilter_inout_alloc();
        AVFilterInOut* inputs = ffmpeg.avfilter_inout_alloc();
        outputs->name = ffmpeg.av_strdup("in");
        outputs->filter_ctx = _source;
        outputs->pad_idx = 0;
        outputs->next = null;
        inputs->name = ffmpeg.av_strdup("out");
        inputs->filter_ctx = _sink;
        inputs->pad_idx = 0;
        inputs->next = null;
        try
        {
            string graph = $"split[a][b];[a]palettegen=stats_mode=full[p];[b][p]paletteuse=dither={dither}:diff_mode=rectangle";
            FfmpegError.Check(ffmpeg.avfilter_graph_parse_ptr(_graph, graph, &inputs, &outputs, null), "build the palette filter");
            FfmpegError.Check(ffmpeg.avfilter_graph_config(_graph, null), "configure the palette filter");
        }
        finally
        {
            ffmpeg.avfilter_inout_free(&inputs);
            ffmpeg.avfilter_inout_free(&outputs);
        }
    }

    /// <summary>Queues one RGB frame (copied by the filter).</summary>
    /// <param name="rgb">The pixels, <c>width * height * 3</c> bytes.</param>
    /// <param name="index">The frame's timestamp in frames.</param>
    public void Add(byte[] rgb, long index)
    {
        fixed (byte* pixels = rgb)
        {
            _input->format = (int)AVPixelFormat.AV_PIX_FMT_RGB24;
            _input->width = _width;
            _input->height = _height;
            _input->data[0] = pixels;
            _input->linesize[0] = _width * 3;
            _input->pts = index;
            FfmpegError.Check(ffmpeg.av_buffersrc_add_frame_flags(_source, _input, (int)AvBuffersrcFlag.AV_BUFFERSRC_FLAG_KEEP_REF), "filter a frame");
            _input->data[0] = null;
        }
    }

    /// <summary>Ends the input and hands every indexed frame to <paramref name="consume"/>.</summary>
    /// <param name="consume">Receives each output frame (valid only during the call).</param>
    public void Finish(FrameConsumer consume)
    {
        FfmpegError.Check(ffmpeg.av_buffersrc_add_frame_flags(_source, null, 0), "end the filter input");
        while (true)
        {
            int result = ffmpeg.av_buffersink_get_frame(_sink, _output);
            if (FfmpegError.IsAgainOrEnd(result))
            {
                return;
            }

            FfmpegError.Check(result, "filter a frame");
            consume(_output);
            ffmpeg.av_frame_unref(_output);
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        AVFrame* input = _input;
        ffmpeg.av_frame_free(&input);
        AVFrame* output = _output;
        ffmpeg.av_frame_free(&output);
        _input = null;
        _output = null;
        if (_graph is not null)
        {
            AVFilterGraph* graph = _graph;
            ffmpeg.avfilter_graph_free(&graph);
            _graph = null;
        }
    }

    /// <summary>Receives a filtered frame.</summary>
    /// <param name="frame">The frame, valid only during the call.</param>
    internal delegate void FrameConsumer(AVFrame* frame);
}
