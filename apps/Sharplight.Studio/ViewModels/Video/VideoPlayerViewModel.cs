using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Vision;

namespace Sharplight.Studio.ViewModels.Video;

/// <summary>
/// The playback control of the Video page: the frames of a video (or the frames decoded so far), a frame position that a scrubber
/// moves, play and pause, and looping. A <see cref="DispatcherTimer"/> advances the position at the video's frame rate; it only runs on
/// the UI thread, so frames are added and the position changes there.
/// </summary>
internal sealed partial class VideoPlayerViewModel : ObservableObject
{
    private const double FallbackFramesPerSecond = 16;

    private IReadOnlyList<ImageInput> _frames = [];
    private DispatcherTimer? _timer;

    /// <summary>The frames shown, in order.</summary>
    public IReadOnlyList<ImageInput> Frames => _frames;

    /// <summary>Frames per second of the playback.</summary>
    [ObservableProperty]
    public partial double FramesPerSecond { get; private set; } = FallbackFramesPerSecond;

    /// <summary>The position of the frame shown (0-based); the scrubber binds to it.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CurrentFrame), nameof(PositionText))]
    public partial double Position { get; set; }

    /// <summary>Playback restarts at the first frame after the last one.</summary>
    [ObservableProperty]
    public partial bool Loop { get; set; } = true;

    /// <summary>The video is playing.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(PlayIcon), nameof(PlayTip))]
    public partial bool IsPlaying { get; private set; }

    /// <summary>The frame at <see cref="Position"/>, or null without frames.</summary>
    public ImageInput? CurrentFrame
    {
        get
        {
            int index = (int)Math.Round(Position);
            return index >= 0 && index < _frames.Count ? _frames[index] : null;
        }
    }

    /// <summary>The last position, for the scrubber's range.</summary>
    public double LastPosition => Math.Max(1, _frames.Count - 1);

    /// <summary>There are frames to show.</summary>
    public bool HasFrames => _frames.Count > 0;

    /// <summary>More than one frame: playing and scrubbing make sense.</summary>
    public bool CanPlay => _frames.Count > 1;

    /// <summary>The icon resource of the play button.</summary>
    public string PlayIcon => IsPlaying ? "IconPause" : "IconPlay";

    /// <summary>The tooltip of the play button.</summary>
    public string PlayTip => IsPlaying ? "Pause" : "Play";

    /// <summary>The position as text (frame of count, seconds).</summary>
    public string PositionText
    {
        get
        {
            if (_frames.Count == 0)
            {
                return "";
            }

            int index = (int)Math.Round(Position);
            double seconds = index / FramesPerSecond;
            double total = _frames.Count / FramesPerSecond;
            return $"Frame {index + 1} of {_frames.Count} · {seconds:0.0} / {total:0.0} s";
        }
    }

    /// <summary>Shows <paramref name="frames"/> from the first one, stopped.</summary>
    /// <param name="frames">The frames of a finished video.</param>
    /// <param name="framesPerSecond">Its frame rate.</param>
    public void Load(IReadOnlyList<ImageInput> frames, double framesPerSecond)
    {
        Pause();
        _frames = frames;
        FramesPerSecond = framesPerSecond > 0 ? framesPerSecond : FallbackFramesPerSecond;
        Position = 0;
        NotifyFramesChanged();
    }

    /// <summary>Shows the last of <paramref name="frames"/> (a preview, or the newest decoded frames) and keeps them as the frames to scrub.</summary>
    /// <param name="frames">The frames so far.</param>
    public void Follow(IReadOnlyList<ImageInput> frames)
    {
        Pause();
        _frames = frames;
        Position = Math.Max(0, frames.Count - 1);
        NotifyFramesChanged();
    }

    /// <summary>Removes the frames.</summary>
    public void Clear()
    {
        Pause();
        _frames = [];
        Position = 0;
        NotifyFramesChanged();
    }

    /// <summary>Advances one frame (what the timer does); stops at the end unless looping.</summary>
    public void Advance()
    {
        if (_frames.Count == 0)
        {
            return;
        }

        int next = (int)Math.Round(Position) + 1;
        if (next >= _frames.Count)
        {
            if (!Loop)
            {
                Pause();
                return;
            }

            next = 0;
        }

        Position = next;
    }

    /// <summary>Starts playing (from the first frame when the position is at the end).</summary>
    public void Play()
    {
        if (!CanPlay)
        {
            return;
        }

        if ((int)Math.Round(Position) >= _frames.Count - 1)
        {
            Position = 0;
        }

        _timer ??= new DispatcherTimer();
        _timer.Tick -= OnTick;
        _timer.Tick += OnTick;
        _timer.Interval = TimeSpan.FromSeconds(1 / FramesPerSecond);
        _timer.Start();
        IsPlaying = true;
    }

    /// <summary>Stops playing.</summary>
    public void Pause()
    {
        _timer?.Stop();
        IsPlaying = false;
    }

    [RelayCommand]
    private void TogglePlay()
    {
        if (IsPlaying)
        {
            Pause();
        }
        else
        {
            Play();
        }
    }

    private void OnTick(object? sender, EventArgs e) => Advance();

    private void NotifyFramesChanged()
    {
        OnPropertyChanged(nameof(Frames));
        OnPropertyChanged(nameof(LastPosition));
        OnPropertyChanged(nameof(HasFrames));
        OnPropertyChanged(nameof(CanPlay));
        OnPropertyChanged(nameof(CurrentFrame));
        OnPropertyChanged(nameof(PositionText));
    }
}
