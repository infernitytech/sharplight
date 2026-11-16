using System.Globalization;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Media;
using Avalonia.Media.Imaging;

namespace Sharplight.Studio.Controls;

/// <summary>
/// Shows one bitmap that can be zoomed (mouse wheel and pinch around the pointer, keys) and panned (drag). It starts fitted to the
/// control, and a double click toggles between fit and 100 %. The bitmap is decoded elsewhere (<see cref="Sharplight.Studio.Converters.DecodedBitmaps"/>),
/// so drawing is a single transformed blit and never decodes on the UI thread.
/// </summary>
internal sealed class ZoomPanView : Control
{
    /// <summary>The factor of one zoom step (buttons, keys, one wheel notch).</summary>
    public const double Step = 1.25;

    /// <summary>The bitmap shown.</summary>
    public static readonly StyledProperty<IImage?> SourceProperty = AvaloniaProperty.Register<ZoomPanView, IImage?>(nameof(Source));

    /// <summary>The zoom, in screen pixels per image pixel (read only).</summary>
    public static readonly DirectProperty<ZoomPanView, double> ZoomProperty = AvaloniaProperty.RegisterDirect<ZoomPanView, double>(nameof(Zoom), view => view._transform.Scale);

    /// <summary>The zoom as text (<c>fit 38 %</c>, <c>100 %</c>) for a label.</summary>
    public static readonly DirectProperty<ZoomPanView, string> ZoomTextProperty = AvaloniaProperty.RegisterDirect<ZoomPanView, string>(nameof(ZoomText), view => view._zoomText);

    private ViewTransform _transform = new(1, default);
    private bool _fitted = true;
    private string _zoomText = "";
    private bool _dragging;
    private Point _dragStart;
    private double _pinchScale = 1;

    static ZoomPanView()
    {
        AffectsRender<ZoomPanView>(SourceProperty);
        FocusableProperty.OverrideDefaultValue<ZoomPanView>(true);
        ClipToBoundsProperty.OverrideDefaultValue<ZoomPanView>(true);
    }

    /// <summary>Creates the view with a pinch recognizer.</summary>
    public ZoomPanView()
    {
        GestureRecognizers.Add(new PinchGestureRecognizer());
        AddHandler(PinchEvent, OnPinch);
        AddHandler(PinchEndedEvent, (_, _) => _pinchScale = 1);
    }

    /// <summary>The bitmap shown.</summary>
    public IImage? Source
    {
        get => GetValue(SourceProperty);
        set => SetValue(SourceProperty, value);
    }

    /// <summary>The zoom, in screen pixels per image pixel.</summary>
    public double Zoom => _transform.Scale;

    /// <summary>The zoom as text.</summary>
    public string ZoomText => _zoomText;

    /// <summary>The view is fitted (follows the control's size) rather than zoomed by the user.</summary>
    public bool IsFitted => _fitted;

    private Size ImageSize => Source?.Size ?? default;

    /// <summary>Shows the whole image.</summary>
    public void FitToView()
    {
        _fitted = true;
        Apply(ViewTransform.Fit(ImageSize, Bounds.Size));
    }

    /// <summary>Shows the image at 100 %.</summary>
    public void ShowActualSize()
    {
        _fitted = false;
        Apply(ViewTransform.ActualSize(ImageSize, Bounds.Size));
    }

    /// <summary>Zooms in one step around the centre.</summary>
    public void ZoomIn() => ZoomBy(Step, new Point(Bounds.Width / 2, Bounds.Height / 2));

    /// <summary>Zooms out one step around the centre.</summary>
    public void ZoomOut() => ZoomBy(1 / Step, new Point(Bounds.Width / 2, Bounds.Height / 2));

    /// <summary>Toggles between fit and 100 %.</summary>
    public void ToggleFit()
    {
        if (_fitted && ViewTransform.FitScale(ImageSize, Bounds.Size) < 1)
        {
            ShowActualSize();
        }
        else
        {
            FitToView();
        }
    }

    /// <summary>Zooms by <paramref name="factor"/> around a viewport point.</summary>
    public void ZoomBy(double factor, Point anchor)
    {
        if (ImageSize.Width <= 0)
        {
            return;
        }

        _fitted = false;
        Apply(_transform.ZoomAround(anchor, factor, ImageSize, Bounds.Size));
    }

    /// <inheritdoc/>
    protected override void OnPropertyChanged(AvaloniaPropertyChangedEventArgs change)
    {
        base.OnPropertyChanged(change);
        if (change.Property == SourceProperty)
        {
            var old = change.OldValue as IImage;
            var now = change.NewValue as IImage;
            // The same size keeps the view when the picture is replaced by another version of itself (a live preview step).
            if (old is null || now is null || old.Size != now.Size)
            {
                _fitted = true;
            }

            Apply(_fitted ? ViewTransform.Fit(ImageSize, Bounds.Size) : _transform.Clamped(ImageSize, Bounds.Size));
        }
        else if (change.Property == BoundsProperty)
        {
            Apply(_fitted ? ViewTransform.Fit(ImageSize, Bounds.Size) : _transform.Clamped(ImageSize, Bounds.Size));
        }
    }

    private void Apply(ViewTransform transform)
    {
        double oldScale = _transform.Scale;
        string oldText = _zoomText;
        _transform = transform;
        _zoomText = ImageSize.Width <= 0 ? "" : (_fitted ? "Fit " : "") + (transform.Scale * 100).ToString("0", CultureInfo.InvariantCulture) + " %";
        if (oldScale != transform.Scale)
        {
            RaisePropertyChanged(ZoomProperty, oldScale, transform.Scale);
        }

        if (oldText != _zoomText)
        {
            RaisePropertyChanged(ZoomTextProperty, oldText, _zoomText);
        }

        InvalidateVisual();
    }

    /// <inheritdoc/>
    public override void Render(DrawingContext context)
    {
        if (Source is not { } source || ImageSize.Width <= 0)
        {
            return;
        }

        var target = new Rect(_transform.Offset.X, _transform.Offset.Y, ImageSize.Width * _transform.Scale, ImageSize.Height * _transform.Scale);
        // Single pixels are shown as squares when zoomed in far; a good filter when the picture is reduced.
        var mode = _transform.Scale >= 3 ? BitmapInterpolationMode.None : _transform.Scale < 1 ? BitmapInterpolationMode.HighQuality : BitmapInterpolationMode.MediumQuality;
        using (context.PushRenderOptions(new RenderOptions { BitmapInterpolationMode = mode }))
        {
            context.DrawImage(source, new Rect(ImageSize), target);
        }
    }

    /// <inheritdoc/>
    protected override void OnPointerWheelChanged(PointerWheelEventArgs e)
    {
        base.OnPointerWheelChanged(e);
        if (e.Delta.Y != 0)
        {
            ZoomBy(Math.Pow(Step, Math.Clamp(e.Delta.Y, -3, 3)), e.GetPosition(this));
            e.Handled = true;
        }
    }

    /// <inheritdoc/>
    protected override void OnPointerPressed(PointerPressedEventArgs e)
    {
        base.OnPointerPressed(e);
        Focus();
        if (e.ClickCount == 2 && e.GetCurrentPoint(this).Properties.IsLeftButtonPressed)
        {
            ToggleFit();
            e.Handled = true;
        }
        else if (e.Pointer.Type == PointerType.Mouse && e.GetCurrentPoint(this).Properties.IsLeftButtonPressed || e.GetCurrentPoint(this).Properties.IsMiddleButtonPressed)
        {
            _dragging = true;
            _dragStart = e.GetPosition(this);
            e.Pointer.Capture(this);
            Cursor = new Cursor(StandardCursorType.SizeAll);
            e.Handled = true;
        }
    }

    /// <inheritdoc/>
    protected override void OnPointerMoved(PointerEventArgs e)
    {
        base.OnPointerMoved(e);
        if (_dragging)
        {
            var position = e.GetPosition(this);
            _fitted = false;
            Apply(_transform.Pan(position - _dragStart, ImageSize, Bounds.Size));
            _dragStart = position;
        }
    }

    /// <inheritdoc/>
    protected override void OnPointerReleased(PointerReleasedEventArgs e)
    {
        base.OnPointerReleased(e);
        if (_dragging)
        {
            _dragging = false;
            e.Pointer.Capture(null);
            Cursor = null;
        }
    }

    /// <inheritdoc/>
    protected override void OnKeyDown(KeyEventArgs e)
    {
        base.OnKeyDown(e);
        switch (e.Key)
        {
            case Key.Add or Key.OemPlus:
                ZoomIn();
                break;
            case Key.Subtract or Key.OemMinus:
                ZoomOut();
                break;
            case Key.D0 or Key.NumPad0:
                FitToView();
                break;
            case Key.D1 or Key.NumPad1:
                ShowActualSize();
                break;
            case Key.Left:
                PanBy(new Vector(48, 0));
                break;
            case Key.Right:
                PanBy(new Vector(-48, 0));
                break;
            case Key.Up:
                PanBy(new Vector(0, 48));
                break;
            case Key.Down:
                PanBy(new Vector(0, -48));
                break;
            default:
                return;
        }

        e.Handled = true;
    }

    private void PanBy(Vector delta)
    {
        _fitted = false;
        Apply(_transform.Pan(delta, ImageSize, Bounds.Size));
    }

    private void OnPinch(object? sender, PinchEventArgs e)
    {
        double factor = e.Scale / _pinchScale;
        _pinchScale = e.Scale;
        if (factor > 0 && !double.IsInfinity(factor))
        {
            ZoomBy(factor, e.ScaleOrigin);
        }
    }
}
