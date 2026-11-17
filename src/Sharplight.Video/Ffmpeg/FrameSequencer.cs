namespace Sharplight.Video.Ffmpeg;

/// <summary>
/// Numbers decoded frames and, with a target frame rate, resamples them: each output slot <c>start + k / rate</c> shows the last frame
/// whose presentation time is at or before it (frames are repeated or dropped).
/// </summary>
internal sealed class FrameSequencer
{
    private const double Epsilon = 1e-6;
    private readonly double? _rate;
    private readonly double _start;
    private readonly int _limit;
    private readonly Action<VideoFrame> _emit;
    private VideoFrame? _previous;
    private int _count;

    /// <param name="rate">The output frame rate, or <see langword="null"/> to pass every frame through.</param>
    /// <param name="start">The time of the first output slot in seconds.</param>
    /// <param name="limit">The most frames to emit.</param>
    /// <param name="emit">Receives the numbered frames.</param>
    public FrameSequencer(double? rate, double start, int limit, Action<VideoFrame> emit)
    {
        if (rate is double value && (!double.IsFinite(value) || value <= 0))
        {
            throw new ArgumentOutOfRangeException(nameof(rate), value, "The frame rate must be positive and finite.");
        }

        _rate = rate;
        _start = start;
        _limit = limit;
        _emit = emit;
    }

    /// <summary>Whether the frame limit is reached.</summary>
    public bool IsFull => _count >= _limit;

    /// <summary>Accepts the next decoded frame.</summary>
    /// <param name="frame">The converted frame (its index and timestamp are replaced).</param>
    /// <param name="time">Its presentation time in seconds.</param>
    public void Add(VideoFrame frame, double time)
    {
        if (_rate is null)
        {
            Emit(frame, time);
            return;
        }

        // Slots before this frame show the previous one (or, before the first frame, this one).
        EmitSlotsBefore(time, _previous ?? frame);
        _previous = frame;
    }

    /// <summary>Fills the slots up to the end of the last frame.</summary>
    /// <param name="end">Where the last frame stops showing, in seconds.</param>
    public void Finish(double end)
    {
        if (_rate is not null && _previous is not null)
        {
            EmitSlotsBefore(end, _previous);
        }
    }

    private void EmitSlotsBefore(double time, VideoFrame frame)
    {
        double rate = _rate!.Value;
        while (!IsFull)
        {
            double slot = _start + _count / rate;
            if (slot >= time - Epsilon)
            {
                return;
            }

            Emit(frame, slot);
        }
    }

    private void Emit(VideoFrame frame, double time)
    {
        if (IsFull)
        {
            return;
        }

        _emit(frame with { Index = _count, Timestamp = TimeSpan.FromSeconds(Math.Max(0, time)) });
        _count++;
    }
}
