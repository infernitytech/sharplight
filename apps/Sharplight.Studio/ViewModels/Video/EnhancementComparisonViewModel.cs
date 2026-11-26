using System.ComponentModel;
using System.Globalization;
using CommunityToolkit.Mvvm.ComponentModel;
using Sharplight.Studio.Services.Imaging;
using Sharplight.Studio.Services.Platform;
using Sharplight.Vision;

namespace Sharplight.Studio.ViewModels.Video;

/// <summary>
/// The live before/after comparison of the Enhance page: every finished enhanced frame paired with the input frame of the same index,
/// both at one preview size (the input enlarged with plain bilinear) so they overlay exactly. A blend slider crossfades from the input (0)
/// to the result (1), or, in split mode, moves a vertical divider (input left, result right). While a run streams batches the newest
/// frame is shown; the frames finished so far can be scrubbed and played (looping) through <see cref="Player"/>.
/// </summary>
/// <remarks>
/// Memory: frames are resized on the thread pool (one batch after another) and only the previews are kept, never extra full-size copies.
/// The preview size is chosen when the first batch arrives so that the expected frames, at most <see cref="PreviewWindowFrames"/> of them,
/// fit <see cref="MemoryBudgetBytes"/> (the model's output size when they fit, else both layers reduced alike, never below
/// <see cref="MinPreviewSide"/> pixels on the short side). When more frames arrive than fit the budget, the oldest are let go and the
/// newest kept, so a clip of any length holds at most the budget.
/// </remarks>
internal sealed partial class EnhancementComparisonViewModel : ObservableObject
{
    /// <summary>The default memory cap of the kept frames (both layers).</summary>
    public const long DefaultMemoryBudgetBytes = 1L << 30;

    /// <summary>The smallest short side the preview is reduced to.</summary>
    public const int MinPreviewSide = 180;

    /// <summary>The most frames the preview size is chosen for: a long clip keeps its newest frames at a useful size rather than all of them tiny.</summary>
    public const int PreviewWindowFrames = 240;

    private const int BytesPerPixelPair = 6;

    private readonly IUiDispatcher _dispatcher;
    private readonly object _gate = new();
    private readonly List<ComparisonFrame> _frames = [];
    private int _generation;
    private long _expectedFrames = 1;
    private (int Width, int Height)? _previewSize;
    private (int Width, int Height) _outputSize;
    private Task _chain = Task.CompletedTask;
    private Task _idle = Task.CompletedTask;
    private long _dropped;

    /// <summary>Creates the comparison.</summary>
    /// <param name="dispatcher">Runs the updates of the shown frames on the UI thread.</param>
    public EnhancementComparisonViewModel(IUiDispatcher dispatcher)
    {
        _dispatcher = dispatcher;
        Player.PropertyChanged += OnPlayerChanged;
    }

    /// <summary>The scrubber and play/pause over the frames kept (its frames are the enhanced previews, in order).</summary>
    public VideoPlayerViewModel Player { get; } = new();

    /// <summary>The cap of the bytes kept for both layers; read when a run's first batch arrives.</summary>
    public long MemoryBudgetBytes { get; set; } = DefaultMemoryBudgetBytes;

    /// <summary>The slider: the opacity of the result over the input (0 = input, 1 = result), or the divider position in split mode.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(EnhancedOpacity), nameof(BlendText))]
    public partial double Blend { get; set; } = 1;

    /// <summary>The slider moves a vertical divider (input left, result right) instead of crossfading.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(EnhancedOpacity), nameof(BlendText))]
    public partial bool IsSplit { get; set; }

    /// <summary>A new batch moves the view to its newest frame (unless playing).</summary>
    [ObservableProperty]
    public partial bool FollowNewest { get; set; } = true;

    /// <summary>The frame pair shown, or null.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(PositionText))]
    public partial ComparisonFrame? Current { get; private set; }

    /// <summary>What the kept frames are compared at (when reduced to fit the memory cap), or empty.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasPreviewNote))]
    public partial string PreviewNote { get; private set; } = "";

    /// <summary>There is a <see cref="PreviewNote"/>.</summary>
    public bool HasPreviewNote => PreviewNote.Length > 0;

    /// <summary>The opacity of the result layer: the blend when crossfading, 1 in split mode (the divider clips it instead).</summary>
    public double EnhancedOpacity => IsSplit ? 1 : Math.Clamp(Blend, 0, 1);

    /// <summary>The slider's meaning and value.</summary>
    public string BlendText => IsSplit
        ? $"Divider {Math.Clamp(Blend, 0, 1) * 100:0}%"
        : $"Enhanced {Math.Clamp(Blend, 0, 1) * 100:0}%";

    /// <summary>The frames kept.</summary>
    public int Count => _frames.Count;

    /// <summary>There is a frame to compare.</summary>
    public bool HasFrames => _frames.Count > 0;

    /// <summary>The frames kept, oldest first.</summary>
    public IReadOnlyList<ComparisonFrame> Frames => _frames;

    /// <summary>The bytes the kept frames hold.</summary>
    public long StoredBytes { get; private set; }

    /// <summary>The frames finished in this run (kept or let go).</summary>
    public long FinishedFrames => _dropped + _frames.Count;

    /// <summary>The shown frame and how far the run is.</summary>
    public string PositionText => Current is not { } frame
        ? ""
        : string.Create(CultureInfo.InvariantCulture, $"Frame {frame.Index + 1} · {FinishedFrames} of {Math.Max(_expectedFrames, FinishedFrames)} finished");

    /// <summary>Completes when every batch handed over so far is resized and shown (for tests).</summary>
    public Task Idle
    {
        get
        {
            lock (_gate)
            {
                return _idle;
            }
        }
    }

    /// <summary>The preview size of frames of <paramref name="width"/> × <paramref name="height"/> when <paramref name="frames"/> of them must fit <paramref name="budget"/> bytes (both layers).</summary>
    /// <param name="width">The enhanced frames' width.</param>
    /// <param name="height">The enhanced frames' height.</param>
    /// <param name="frames">The frames expected.</param>
    /// <param name="budget">The memory cap in bytes.</param>
    internal static (int Width, int Height) PreviewSizeFor(int width, int height, long frames, long budget)
    {
        double needed = (double)Math.Max(1, frames) * width * height * BytesPerPixelPair;
        if (needed <= budget)
        {
            return (width, height);
        }

        double scale = Math.Sqrt(budget / needed);
        double floor = Math.Min(1, (double)MinPreviewSide / Math.Min(width, height));
        scale = Math.Max(scale, floor);
        return (Math.Max(1, (int)Math.Round(width * scale)), Math.Max(1, (int)Math.Round(height * scale)));
    }

    /// <summary>Starts the comparison of a new run (forgets the frames shown). Callable from any thread.</summary>
    /// <param name="expectedFrames">The frames the run will produce (sizes the previews; an estimate is fine).</param>
    /// <param name="framesPerSecond">The playback rate.</param>
    public void Begin(long expectedFrames, double framesPerSecond)
    {
        int generation;
        lock (_gate)
        {
            generation = ++_generation;
            _expectedFrames = Math.Max(1, expectedFrames);
            _previewSize = null;
        }

        _dispatcher.Post(() =>
        {
            if (IsCurrent(generation))
            {
                Reset(framesPerSecond);
            }
        });
    }

    /// <summary>Adds the finished frames of a batch, paired with their input frames. Callable from any thread; the resizing runs on the thread pool.</summary>
    /// <param name="firstIndex">The clip index of <paramref name="enhanced"/>[0].</param>
    /// <param name="originals">The input frames of the batch, in the same order (a single frame pairs with all of them, as for a still image); not modified.</param>
    /// <param name="enhanced">The batch's finished frames, in order.</param>
    /// <returns>Completes when the batch is resized and shown; a caller that waits for it before adding the next batch never queues more than one.</returns>
    public Task Add(long firstIndex, IReadOnlyList<ImageInput> originals, IReadOnlyList<ImageInput> enhanced)
    {
        if (enhanced.Count == 0 || originals.Count == 0)
        {
            return Task.CompletedTask;
        }

        var done = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        lock (_gate)
        {
            int generation = _generation;
            _chain = _chain.ContinueWith(_ => Prepare(generation, firstIndex, originals, enhanced, done), CancellationToken.None,
                TaskContinuationOptions.None, TaskScheduler.Default);
            _idle = done.Task;
        }

        return done.Task;
    }

    /// <summary>Forgets the frames (a new input, or a restored run). UI thread.</summary>
    public void Clear()
    {
        lock (_gate)
        {
            _generation++;
            _expectedFrames = 1;
            _previewSize = null;
        }

        Reset(Player.FramesPerSecond);
    }

    private bool IsCurrent(int generation)
    {
        lock (_gate)
        {
            return generation == _generation;
        }
    }

    /// <summary>Resizes a batch to the preview size (thread pool) and hands it to the UI thread.</summary>
    private void Prepare(int generation, long firstIndex, IReadOnlyList<ImageInput> originals, IReadOnlyList<ImageInput> enhanced, TaskCompletionSource done)
    {
        bool handedOver = false;
        try
        {
            (int Width, int Height) size;
            lock (_gate)
            {
                if (generation != _generation)
                {
                    return;
                }

                _outputSize = (enhanced[0].Width, enhanced[0].Height);
                _previewSize ??= PreviewSizeFor(enhanced[0].Width, enhanced[0].Height, Math.Min(_expectedFrames, PreviewWindowFrames), MemoryBudgetBytes);
                size = _previewSize.Value;
            }

            var frames = new ComparisonFrame[enhanced.Count];
            for (int i = 0; i < frames.Length; i++)
            {
                long index = firstIndex + i;
                var original = originals[Math.Min(i, originals.Count - 1)];
                frames[i] = new ComparisonFrame(index, FrameResampler.Resize(original, size.Width, size.Height),
                    FrameResampler.Resize(enhanced[i], size.Width, size.Height));
            }

            _dispatcher.Post(() =>
            {
                try
                {
                    if (IsCurrent(generation))
                    {
                        Append(frames);
                    }
                }
                finally
                {
                    done.TrySetResult();
                }
            });
            handedOver = true;
        }
        finally
        {
            if (!handedOver)
            {
                done.TrySetResult();
            }
        }
    }

    private void Append(ComparisonFrame[] frames)
    {
        _frames.AddRange(frames);
        StoredBytes += frames.Sum(frame => frame.Bytes);
        int drop = 0;
        while (StoredBytes > MemoryBudgetBytes && _frames.Count - drop > 1)
        {
            StoredBytes -= _frames[drop].Bytes;
            drop++;
        }

        if (drop > 0)
        {
            _frames.RemoveRange(0, drop);
            _dropped += drop;
        }

        double position = FollowNewest && !Player.IsPlaying ? _frames.Count - 1 : Player.Position - drop;
        Player.Extend([.. _frames.Select(frame => frame.Enhanced)], position);
        UpdateCurrent();
        UpdatePreviewNote();
        NotifyFramesChanged();
    }

    private void Reset(double framesPerSecond)
    {
        _frames.Clear();
        StoredBytes = 0;
        _dropped = 0;
        Player.Load([], framesPerSecond);
        PreviewNote = "";
        UpdateCurrent();
        NotifyFramesChanged();
    }

    private void UpdatePreviewNote()
    {
        (int Width, int Height)? preview;
        (int Width, int Height) output;
        lock (_gate)
        {
            preview = _previewSize;
            output = _outputSize;
        }

        string budget = string.Create(CultureInfo.InvariantCulture, $"{MemoryBudgetBytes / (1024.0 * 1024):0} MB");
        var notes = new List<string>();
        if (preview is { } size && size != output)
        {
            notes.Add($"Compared at {size.Width} × {size.Height} (the result is {output.Width} × {output.Height}) to keep the frames within {budget}.");
        }

        if (_dropped > 0)
        {
            notes.Add($"The first {_dropped} frames were let go to keep the comparison within {budget}.");
        }

        PreviewNote = string.Join(" ", notes);
    }

    private void OnPlayerChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName is nameof(VideoPlayerViewModel.CurrentFrame))
        {
            UpdateCurrent();
        }
    }

    private void UpdateCurrent()
    {
        int index = (int)Math.Round(Player.Position);
        Current = index >= 0 && index < _frames.Count ? _frames[index] : null;
    }

    private void NotifyFramesChanged()
    {
        OnPropertyChanged(nameof(Count));
        OnPropertyChanged(nameof(HasFrames));
        OnPropertyChanged(nameof(Frames));
        OnPropertyChanged(nameof(FinishedFrames));
        OnPropertyChanged(nameof(PositionText));
    }
}
