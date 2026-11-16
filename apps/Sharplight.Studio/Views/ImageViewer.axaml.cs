using Avalonia;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Media;

namespace Sharplight.Studio.Views;

/// <summary>A zoomable, pannable image (<see cref="Controls.ZoomPanView"/>) with fit, 100 % and zoom buttons; shared by the pages that show pictures.</summary>
internal sealed partial class ImageViewer : UserControl
{
    /// <summary>The bitmap shown.</summary>
    public static readonly StyledProperty<IImage?> SourceProperty = AvaloniaProperty.Register<ImageViewer, IImage?>(nameof(Source));

    /// <summary>Whether there is a bitmap (shows the buttons).</summary>
    public static readonly DirectProperty<ImageViewer, bool> HasImageProperty = AvaloniaProperty.RegisterDirect<ImageViewer, bool>(nameof(HasImage), viewer => viewer.Source is not null);

    /// <summary>The second picture of a before/after comparison (shown right of the divider), or null.</summary>
    public static readonly StyledProperty<IImage?> CompareSourceProperty = AvaloniaProperty.Register<ImageViewer, IImage?>(nameof(CompareSource));

    /// <summary>Where the comparison divides the pictures, 0 to 1.</summary>
    public static readonly StyledProperty<double> DividerProperty = AvaloniaProperty.Register<ImageViewer, double>(nameof(Divider), 0.5, defaultBindingMode: Avalonia.Data.BindingMode.TwoWay);

    /// <summary>The primary pointer button places points and boxes instead of dragging (see <see cref="Controls.ZoomPanView.CapturesPrimaryProperty"/>).</summary>
    public static readonly StyledProperty<bool> CapturesPrimaryProperty = AvaloniaProperty.Register<ImageViewer, bool>(nameof(CapturesPrimary));

    /// <summary>Whether a comparison is shown.</summary>
    public static readonly DirectProperty<ImageViewer, bool> IsComparingProperty = AvaloniaProperty.RegisterDirect<ImageViewer, bool>(nameof(IsComparing), viewer => viewer.CompareSource is not null);

    /// <summary>Raised on a press on the picture while <see cref="CapturesPrimary"/> (position as fractions of the picture).</summary>
    public event EventHandler<Controls.ImagePointerEventArgs>? ImagePressed;

    /// <summary>Raised on the release after <see cref="ImagePressed"/>.</summary>
    public event EventHandler<Controls.ImagePointerEventArgs>? ImageReleased;

    /// <summary>Creates the viewer.</summary>
    public ImageViewer()
    {
        InitializeComponent();
        Zoom.ImagePressed += (_, args) => ImagePressed?.Invoke(this, args);
        Zoom.ImageReleased += (_, args) => ImageReleased?.Invoke(this, args);
        Zoom.PropertyChanged += (_, change) =>
        {
            if (change.Property == Controls.ZoomPanView.ZoomTextProperty)
            {
                ZoomLabel.Text = Zoom.ZoomText;
            }
        };
    }

    /// <summary>The bitmap shown.</summary>
    public IImage? Source
    {
        get => GetValue(SourceProperty);
        set => SetValue(SourceProperty, value);
    }

    /// <summary>The second picture of a comparison.</summary>
    public IImage? CompareSource
    {
        get => GetValue(CompareSourceProperty);
        set => SetValue(CompareSourceProperty, value);
    }

    /// <summary>The comparison divider, 0 to 1.</summary>
    public double Divider
    {
        get => GetValue(DividerProperty);
        set => SetValue(DividerProperty, value);
    }

    /// <summary>The primary button places things instead of dragging.</summary>
    public bool CapturesPrimary
    {
        get => GetValue(CapturesPrimaryProperty);
        set => SetValue(CapturesPrimaryProperty, value);
    }

    /// <summary>Whether a comparison is shown.</summary>
    public bool IsComparing => CompareSource is not null;

    /// <summary>The cursor to show over the picture (a cross while it places things).</summary>
    public Controls.ZoomPanView View => Zoom;

    /// <summary>Whether there is a bitmap.</summary>
    public bool HasImage => Source is not null;

    /// <inheritdoc/>
    protected override void OnPropertyChanged(AvaloniaPropertyChangedEventArgs change)
    {
        base.OnPropertyChanged(change);
        if (change.Property == SourceProperty)
        {
            RaisePropertyChanged(HasImageProperty, change.OldValue is not null, change.NewValue is not null);
        }
        else if (change.Property == CompareSourceProperty)
        {
            RaisePropertyChanged(IsComparingProperty, change.OldValue is not null, change.NewValue is not null);
        }
    }

    private void OnZoomIn(object? sender, RoutedEventArgs e) => Zoom.ZoomIn();

    private void OnZoomOut(object? sender, RoutedEventArgs e) => Zoom.ZoomOut();

    private void OnFit(object? sender, RoutedEventArgs e) => Zoom.FitToView();

    private void OnActualSize(object? sender, RoutedEventArgs e) => Zoom.ShowActualSize();
}
