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

    /// <summary>Creates the viewer.</summary>
    public ImageViewer()
    {
        InitializeComponent();
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
    }

    private void OnZoomIn(object? sender, RoutedEventArgs e) => Zoom.ZoomIn();

    private void OnZoomOut(object? sender, RoutedEventArgs e) => Zoom.ZoomOut();

    private void OnFit(object? sender, RoutedEventArgs e) => Zoom.FitToView();

    private void OnActualSize(object? sender, RoutedEventArgs e) => Zoom.ShowActualSize();
}
