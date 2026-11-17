using System.Diagnostics;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Audio;
using Sharplight.Studio.Services.Audio;
using Sharplight.Vision;

namespace Sharplight.Studio.ViewModels.Video;

/// <summary>
/// The playback control of the Video page: the frames of a video (or the frames decoded so far), a frame position that a scrubber
/// moves, play and pause, and looping. While playing, a <see cref="DispatcherTimer"/> derives the frame from a wall clock (frames are skipped when the UI is late, never
/// slowing the video down); the sound track, when there is one, plays through <see cref="IAudioPlayer"/> on its own thread from the same
/// start time. Drawing the frame (converting it to a bitmap) happens off the UI thread in <see cref="Views.VideoView"/>.
/// </summary>
internal sealed partial class VideoPlayerViewModel : ObservableObject
{
    private const double FallbackFramesPerSecond = 16;

    private readonly IAudioPlayer? _audioPlayer;
    private IReadOnlyList<ImageInput> _frames = [];
    private AudioBuffer? _audio;
    private DispatcherTimer? _timer;
    private readonly Stopwatch _clock = new();
    private double _clockOrigin;
    private bool _ticking;
    private CancellationTokenSource? _audioStop;

    /// <summary>Creates the player.</summary>
    /// <param name="audioPlayer">Plays the clip's sound track; null for silent playback.</param>
    public VideoPlayerViewModel(IAudioPlayer? audioPlayer = null) => _audioPlayer = audioPlayer;

    /// <summary>The frames shown, in order.</summary>
    public IReadOnlyList<ImageInput> Frames => _frames;

    /// <summary>Frames per second of the playback.</summary>
    [ObservableProperty]
    public partial double FramesPerSecond { get; private set; } = FallbackFramesPerSecond;

    /// <summary>The position of the frame shown (0-based); the scrubber binds to it.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CurrentFrame), nameof(PositionText))]
    public partial double Position { get; set; }

    /// <summary>The clip has a sound track that can be played.</summary>
    public bool HasAudio => _audio is not null && _audioPlayer is { IsAvailable: true };

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
    /// <param name="audio">The sound track, or null.</param>
    public void Load(IReadOnlyList<ImageInput> frames, double framesPerSecond, AudioBuffer? audio = null)
    {
        Pause();
        _audio = audio;
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
        _audio = null;
        _frames = frames;
        Position = Math.Max(0, frames.Count - 1);
        NotifyFramesChanged();
    }

    /// <summary>Removes the frames.</summary>
    public void Clear()
    {
        Pause();
        _audio = null;
        _frames = [];
        Position = 0;
        NotifyFramesChanged();
    }

    /// <summary>Advances one frame (a step button); at the end it stops, or wraps to the first frame when looping.</summary>
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

        SetPosition(next);
    }

    /// <summary>Goes back one frame (a step button); at the first frame it wraps to the last one when looping.</summary>
    public void StepBack()
    {
        if (_frames.Count == 0)
        {
            return;
        }

        int previous = (int)Math.Round(Position) - 1;
        if (previous < 0)
        {
            if (!Loop)
            {
                return;
            }

            previous = _frames.Count - 1;
        }

        SetPosition(previous);
    }

    /// <summary>Moves to <paramref name="frame"/> (clamped); playing continues from there.</summary>
    /// <param name="frame">The 0-based frame.</param>
    public void Seek(double frame) => SetPosition(Math.Clamp(Math.Round(frame), 0, Math.Max(0, _frames.Count - 1)));

    /// <summary>Moves to the frame the wall clock says (what each timer tick does); stops at the end unless looping.</summary>
    internal void Tick()
    {
        if (!IsPlaying || _frames.Count == 0)
        {
            return;
        }

        int index = (int)Math.Floor((_clockOrigin + _clock.Elapsed.TotalSeconds) * FramesPerSecond);
        if (index >= _frames.Count)
        {
            if (!Loop)
            {
                _ticking = true;
                Position = _frames.Count - 1;
                _ticking = false;
                Pause();
                return;
            }

            index %= _frames.Count;
            _ticking = true;
            Position = index;
            _ticking = false;
            Rebase();
            return;
        }

        if (index != (int)Math.Round(Position))
        {
            _ticking = true;
            Position = index;
            _ticking = false;
        }
    }

    partial void OnPositionChanged(double value)
    {
        if (IsPlaying && !_ticking)
        {
            Rebase();
        }
    }

    private void SetPosition(double value) => Position = value;

    /// <summary>Restarts the wall clock (and the sound) at the current position, after a seek or a loop.</summary>
    private void Rebase()
    {
        _clockOrigin = Math.Round(Position) / FramesPerSecond;
        _clock.Restart();
        StartAudio();
    }

    private void StartAudio()
    {
        StopAudio();
        if (!HasAudio || _audio is not { } audio || _audioPlayer is null)
        {
            return;
        }

        long first = Math.Min(audio.FrameCount, (long)Math.Round(_clockOrigin * audio.SampleRate));
        if (first >= audio.FrameCount)
        {
            return;
        }

        var stop = new CancellationTokenSource();
        _audioStop = stop;
        var player = _audioPlayer;
        _ = Task.Run(async () =>
        {
            try
            {
                float[] rest = audio.Samples.AsSpan((int)(first * audio.Channels)).ToArray();
                await player.PlayStreamAsync(Single(new AudioBuffer(rest, audio.SampleRate, audio.Channels)), stop.Token).ConfigureAwait(false);
            }
            catch (Exception error) when (error is OperationCanceledException or NotSupportedException or InvalidOperationException or IOException)
            {
                // The sound stops (cancelled, or the output is gone); the picture keeps playing.
            }
        });
    }

    private static async IAsyncEnumerable<AudioBuffer> Single(AudioBuffer buffer)
    {
        yield return buffer;
        await Task.CompletedTask;
    }

    private void StopAudio()
    {
        var stop = Interlocked.Exchange(ref _audioStop, null);
        if (stop is not null)
        {
            stop.Cancel();
            stop.Dispose();
        }
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
        _timer.Interval = TimeSpan.FromSeconds(Math.Clamp(0.5 / FramesPerSecond, 0.004, 0.033));
        IsPlaying = true;
        Rebase();
        _timer.Start();
    }

    /// <summary>Stops playing.</summary>
    public void Pause()
    {
        _timer?.Stop();
        _clock.Stop();
        StopAudio();
        IsPlaying = false;
    }

    [RelayCommand]
    private void StepForward() => Advance();

    [RelayCommand]
    private void StepBackward() => StepBack();

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

    private void OnTick(object? sender, EventArgs e) => Tick();

    private void NotifyFramesChanged()
    {
        OnPropertyChanged(nameof(Frames));
        OnPropertyChanged(nameof(LastPosition));
        OnPropertyChanged(nameof(HasFrames));
        OnPropertyChanged(nameof(CanPlay));
        OnPropertyChanged(nameof(HasAudio));
        OnPropertyChanged(nameof(CurrentFrame));
        OnPropertyChanged(nameof(PositionText));
    }
}
