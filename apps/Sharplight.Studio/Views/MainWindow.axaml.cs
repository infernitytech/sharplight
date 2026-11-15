using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Media;
using Sharplight.Studio.Platform;
using Sharplight.Studio.ViewModels;

namespace Sharplight.Studio.Views;

/// <summary>
/// The main window: title bar, navigation, page host and status bar. With <see cref="WindowChromeMode.Custom"/> it draws its own title
/// bar and resize edges and asks the window manager to move, resize and snap it (so dragging to a screen edge tiles it as usual); with
/// <see cref="WindowChromeMode.System"/> the system frames it.
/// </summary>
internal sealed partial class MainWindow : Window
{
    /// <summary>How far the pointer moves on the title bar before a press becomes a window move (a click or double-click otherwise).</summary>
    private const double DragThreshold = 4;

    private PointerPressedEventArgs? _titleBarPress;
    private Point _titleBarPressPoint;

    public MainWindow()
    {
        InitializeComponent();
        Nav.SelectionChanged += (_, _) =>
        {
            if (DataContext is MainWindowViewModel viewModel && Nav.SelectedItem is PageViewModel page && page != viewModel.CurrentPage)
            {
                _ = viewModel.NavigateAsync(page);
            }
        };

        MinimizeButton.Click += (_, _) => WindowState = WindowState.Minimized;
        MaximizeButton.Click += (_, _) => ToggleMaximized();
        CloseButton.Click += (_, _) => Close();
        TitleBar.PointerPressed += OnTitleBarPressed;
        TitleBar.PointerMoved += OnTitleBarMoved;
        TitleBar.PointerReleased += (_, _) => _titleBarPress = null;
        TitleBar.PointerCaptureLost += (_, _) => _titleBarPress = null;
        foreach (var edge in ResizeEdges.Children.OfType<Border>())
        {
            var side = Enum.Parse<WindowEdge>((string)edge.Tag!);
            edge.PointerPressed += (_, e) =>
            {
                if (e.GetCurrentPoint(edge).Properties.IsLeftButtonPressed)
                {
                    BeginResizeDrag(side, e);
                    e.Handled = true;
                }
            };
        }

        ApplyChrome(WindowChromeMode.Custom);
    }

    /// <summary>The chrome in use.</summary>
    public WindowChromeMode Chrome { get; private set; }

    /// <summary>Switches between the own title bar and the system's.</summary>
    /// <param name="mode">The chrome.</param>
    public void ApplyChrome(WindowChromeMode mode)
    {
        Chrome = mode;
        WindowDecorations = mode == WindowChromeMode.Custom ? WindowDecorations.None : WindowDecorations.Full;
        TitleBar.IsVisible = mode == WindowChromeMode.Custom;
        UpdateFrame();
    }

    /// <inheritdoc/>
    protected override void OnPropertyChanged(AvaloniaPropertyChangedEventArgs change)
    {
        base.OnPropertyChanged(change);
        if (change.Property == WindowStateProperty && TitleBar is not null)
        {
            UpdateFrame();
        }
    }

    private void ToggleMaximized() => WindowState = WindowState == WindowState.Maximized ? WindowState.Normal : WindowState.Maximized;

    /// <summary>Edges and a hairline border while the window is free; neither when maximized or full screen (or framed by the system).</summary>
    private void UpdateFrame()
    {
        bool maximized = WindowState is WindowState.Maximized or WindowState.FullScreen;
        bool framed = Chrome == WindowChromeMode.Custom && !maximized;
        ResizeEdges.IsVisible = framed;
        Frame.BorderThickness = framed ? new Thickness(1) : default;
        MaximizeGlyph.Data = (Geometry)this.FindResource(maximized ? "CaptionRestore" : "CaptionMaximize")!;
        ToolTip.SetTip(MaximizeButton, maximized ? "Restore" : "Maximize");
    }

    private void OnTitleBarPressed(object? sender, PointerPressedEventArgs e)
    {
        if (!e.GetCurrentPoint(TitleBar).Properties.IsLeftButtonPressed)
        {
            return;
        }

        e.Handled = true;
        if (e.ClickCount == 2)
        {
            _titleBarPress = null;
            ToggleMaximized();
            return;
        }

        // The move starts once the pointer travels, so the second click of a double-click still reaches the window.
        _titleBarPress = e;
        _titleBarPressPoint = e.GetPosition(this);
    }

    private void OnTitleBarMoved(object? sender, PointerEventArgs e)
    {
        if (_titleBarPress is not { } press)
        {
            return;
        }

        var delta = e.GetPosition(this) - _titleBarPressPoint;
        if (Math.Abs(delta.X) >= DragThreshold || Math.Abs(delta.Y) >= DragThreshold)
        {
            _titleBarPress = null;
            BeginMoveDrag(press);
        }
    }
}
