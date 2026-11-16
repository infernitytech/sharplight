using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Sharplight.Studio.ViewModels.Vision;

namespace Sharplight.Studio.Views;

/// <summary>The Vision page (classification, embedding, detection, depth, segmentation, text recognition and upscaling). On a segmentation the pointer prompts the model.</summary>
internal sealed partial class VisionPage : UserControl
{
    private Point? _dragStart;

    public VisionPage()
    {
        InitializeComponent();
        var image = this.FindControl<Image>("MainImage")!;
        image.PointerEntered += (_, _) => image.Cursor = ViewModel is { CanPrompt: true } ? new Cursor(StandardCursorType.Cross) : Cursor.Default;
        image.PointerPressed += (_, e) => OnPressed(image, e);
        image.PointerReleased += (_, e) => OnReleased(image, e);
    }

    private VisionPageViewModel? ViewModel => DataContext as VisionPageViewModel;

    private void OnPressed(Image image, PointerPressedEventArgs e)
    {
        if (ViewModel is not { CanPrompt: true } model)
        {
            return;
        }

        var point = e.GetCurrentPoint(image);
        var position = Normalize(image, point.Position);
        if (model.PromptTool == SegmentTool.Box)
        {
            _dragStart = position;
            e.Handled = true;
            return;
        }

        if (point.Properties.IsLeftButtonPressed || point.Properties.IsRightButtonPressed)
        {
            model.PlacePoint(position.X, position.Y, negative: point.Properties.IsRightButtonPressed || e.KeyModifiers.HasFlag(KeyModifiers.Control));
            e.Handled = true;
        }
    }

    private void OnReleased(Image image, PointerReleasedEventArgs e)
    {
        if (_dragStart is not { } start)
        {
            return;
        }

        _dragStart = null;
        if (ViewModel is { CanPrompt: true } model)
        {
            var end = Normalize(image, e.GetCurrentPoint(image).Position);
            model.PlaceBox(start.X, start.Y, end.X, end.Y);
        }
    }

    // The position as fractions of the displayed image: the image control is exactly as large as the picture it shows.
    private static Point Normalize(Image image, Point position)
        => new(Math.Clamp(position.X / Math.Max(1, image.Bounds.Width), 0, 1), Math.Clamp(position.Y / Math.Max(1, image.Bounds.Height), 0, 1));
}
