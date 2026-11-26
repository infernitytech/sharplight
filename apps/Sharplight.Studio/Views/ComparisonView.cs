using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Shapes;
using Avalonia.Layout;
using Avalonia.Media;
using Avalonia.Media.Imaging;
using Avalonia.Threading;
using Sharplight.Studio.ViewModels.Video;
using Sharplight.Vision;

namespace Sharplight.Studio.Views;

/// <summary>
/// The Enhance page's before/after view: the input frame and the enhanced frame of one <see cref="ComparisonFrame"/> as two image layers of
/// the same size, fitted to the same place. The result layer lies on top, either with <see cref="Blend"/> as its opacity (crossfade) or,
/// in <see cref="Split"/> mode, fully opaque and clipped to the right of a vertical divider at <see cref="Blend"/> of the picture's width.
/// Both frames of a pair are converted on the thread pool into one of two pairs of <see cref="WriteableBitmap"/>s (reused) while the other
/// pair is on screen, and swapped together on the UI thread, so the layers never show different frames. A pair that arrives while
/// another is converting replaces the waiting one (frames are dropped, the UI never stalls).
/// </summary>
internal sealed class ComparisonView : UserControl
{
    /// <summary>The frame pair shown, or null.</summary>
    public static readonly StyledProperty<ComparisonFrame?> FrameProperty = AvaloniaProperty.Register<ComparisonView, ComparisonFrame?>(nameof(Frame));

    /// <summary>The result's opacity (crossfade) or the divider position (split), 0 to 1.</summary>
    public static readonly StyledProperty<double> BlendProperty = AvaloniaProperty.Register<ComparisonView, double>(nameof(Blend), 1);

    /// <summary>Whether <see cref="Blend"/> moves a divider instead of fading.</summary>
    public static readonly StyledProperty<bool> SplitProperty = AvaloniaProperty.Register<ComparisonView, bool>(nameof(Split));

    private readonly Image _original = new() { Name = "OriginalLayer", Stretch = Stretch.Uniform };
    private readonly Image _enhanced = new() { Name = "EnhancedLayer", Stretch = Stretch.Uniform };
    private readonly Rectangle _divider = new() { Name = "Divider", Width = 2, Fill = Brushes.White, HorizontalAlignment = HorizontalAlignment.Left, IsVisible = false, IsHitTestVisible = false };
    private readonly WriteableBitmap?[,] _buffers = new WriteableBitmap?[2, 2];
    private readonly object _gate = new();
    private ComparisonFrame? _waiting;
    private bool _converting;
    private int _next;
    private TaskCompletionSource _idle = Completed();

    /// <summary>Creates the view.</summary>
    public ComparisonView()
    {
        RenderOptions.SetBitmapInterpolationMode(_original, BitmapInterpolationMode.LowQuality);
        RenderOptions.SetBitmapInterpolationMode(_enhanced, BitmapInterpolationMode.LowQuality);
        _divider.Effect = new DropShadowEffect { BlurRadius = 4, OffsetX = 0, OffsetY = 0, Opacity = 0.6, Color = Colors.Black };
        var layers = new Grid { ClipToBounds = true };
        layers.Children.Add(_original);
        layers.Children.Add(_enhanced);
        layers.Children.Add(_divider);
        Content = layers;
        UpdateLayers();
    }

    /// <summary>The frame pair shown.</summary>
    public ComparisonFrame? Frame
    {
        get => GetValue(FrameProperty);
        set => SetValue(FrameProperty, value);
    }

    /// <summary>The result's opacity, or the divider position in split mode.</summary>
    public double Blend
    {
        get => GetValue(BlendProperty);
        set => SetValue(BlendProperty, value);
    }

    /// <summary>Whether the slider moves a divider.</summary>
    public bool Split
    {
        get => GetValue(SplitProperty);
        set => SetValue(SplitProperty, value);
    }

    /// <summary>The layer of the input frame.</summary>
    public Image OriginalLayer => _original;

    /// <summary>The layer of the enhanced frame (on top).</summary>
    public Image EnhancedLayer => _enhanced;

    /// <summary>The index of the frame pair on screen, or -1.</summary>
    public long ShownIndex { get; private set; } = -1;

    /// <summary>Completes when every pair handed over so far was converted and put on screen (for tests).</summary>
    public Task Idle
    {
        get
        {
            lock (_gate)
            {
                return _idle.Task;
            }
        }
    }

    /// <inheritdoc/>
    protected override void OnPropertyChanged(AvaloniaPropertyChangedEventArgs change)
    {
        base.OnPropertyChanged(change);
        if (change.Property == FrameProperty)
        {
            Submit(change.GetNewValue<ComparisonFrame?>());
        }
        else if (change.Property == BlendProperty || change.Property == SplitProperty || change.Property == BoundsProperty)
        {
            UpdateLayers();
        }
    }

    /// <summary>Applies the blend: the result's opacity, or its clip and the divider line.</summary>
    private void UpdateLayers()
    {
        double blend = Math.Clamp(Blend, 0, 1);
        if (!Split)
        {
            _enhanced.Opacity = blend;
            _enhanced.Clip = null;
            _divider.IsVisible = false;
            return;
        }

        _enhanced.Opacity = 1;
        Rect picture = PictureRect();
        double x = picture.X + blend * picture.Width;
        _enhanced.Clip = new RectangleGeometry(new Rect(x, 0, Math.Max(0, Bounds.Width - x), Bounds.Height));
        _divider.Margin = new Thickness(x - _divider.Width / 2, picture.Y, 0, 0);
        _divider.Height = picture.Height;
        _divider.VerticalAlignment = VerticalAlignment.Top;
        _divider.IsVisible = _enhanced.Source is not null;
    }

    /// <summary>Where the fitted picture lies inside the view.</summary>
    private Rect PictureRect()
    {
        var size = Bounds.Size;
        if (_enhanced.Source is not { } source || size.Width <= 0 || size.Height <= 0)
        {
            return new Rect(size);
        }

        var pixels = source.Size;
        double scale = Math.Min(size.Width / pixels.Width, size.Height / pixels.Height);
        double width = pixels.Width * scale;
        double height = pixels.Height * scale;
        return new Rect((size.Width - width) / 2, (size.Height - height) / 2, width, height);
    }

    private static TaskCompletionSource Completed()
    {
        var source = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        source.SetResult();
        return source;
    }

    private void Submit(ComparisonFrame? frame)
    {
        if (frame is null)
        {
            lock (_gate)
            {
                _waiting = null;
            }

            _original.Source = null;
            _enhanced.Source = null;
            ShownIndex = -1;
            UpdateLayers();
            return;
        }

        lock (_gate)
        {
            _waiting = frame;
            if (_converting)
            {
                return;
            }

            _converting = true;
            _idle = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        }

        _ = Task.Run(ConvertLoop);
    }

    private void ConvertLoop()
    {
        while (true)
        {
            ComparisonFrame frame;
            TaskCompletionSource idle;
            lock (_gate)
            {
                if (_waiting is null)
                {
                    _converting = false;
                    idle = _idle;
                    Dispatcher.UIThread.Post(() => idle.TrySetResult(), DispatcherPriority.Render);
                    return;
                }

                frame = _waiting;
                _waiting = null;
            }

            WriteableBitmap original;
            WriteableBitmap enhanced;
            try
            {
                int slot = _next;
                _next ^= 1;
                original = Convert(frame.Original, slot, 0);
                enhanced = Convert(frame.Enhanced, slot, 1);
            }
            catch (Exception error) when (error is InvalidOperationException or ArgumentException)
            {
                continue;
            }

            Dispatcher.UIThread.Post(() => Show(frame, original, enhanced), DispatcherPriority.Render);
        }
    }

    private void Show(ComparisonFrame frame, WriteableBitmap original, WriteableBitmap enhanced)
    {
        if (Frame is null)
        {
            return;
        }

        _original.Source = original;
        _enhanced.Source = enhanced;

        // Same-size bitmaps keep their Source reference across swaps of a slot; force a redraw of the new pixels.
        _original.InvalidateVisual();
        _enhanced.InvalidateVisual();
        ShownIndex = frame.Index;
        UpdateLayers();
    }

    private WriteableBitmap Convert(ImageInput frame, int slot, int layer)
    {
        var bitmap = FrameBitmaps.Fill(frame, _buffers[slot, layer]);
        _buffers[slot, layer] = bitmap;
        return bitmap;
    }
}
