using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Sharplight.Studio.Controls;
using Sharplight.Studio.ViewModels.Vision;

namespace Sharplight.Studio.Views;

/// <summary>The Vision page (classification, embedding, detection, depth, segmentation, text recognition and upscaling). On a segmentation the pointer prompts the model.</summary>
internal sealed partial class VisionPage : UserControl
{
    private Point? _dragStart;

    public VisionPage()
    {
        InitializeComponent();
        var viewer = this.FindControl<ImageViewer>("MainViewer")!;
        viewer.View.PointerEntered += (_, _) => viewer.View.Cursor = ViewModel is { CanPrompt: true } ? new Cursor(StandardCursorType.Cross) : Cursor.Default;
        viewer.ImagePressed += OnPressed;
        viewer.ImageReleased += OnReleased;
    }

    private VisionPageViewModel? ViewModel => DataContext as VisionPageViewModel;

    private void OnPressed(object? sender, ImagePointerEventArgs e)
    {
        if (ViewModel is not { CanPrompt: true } model)
        {
            return;
        }

        if (model.PromptTool == SegmentTool.Box)
        {
            _dragStart = e.Position;
            return;
        }

        model.PlacePoint(e.Position.X, e.Position.Y, negative: e.IsSecondary || e.Modifiers.HasFlag(KeyModifiers.Control));
    }

    private void OnReleased(object? sender, ImagePointerEventArgs e)
    {
        if (_dragStart is not { } start)
        {
            return;
        }

        _dragStart = null;
        if (ViewModel is { CanPrompt: true } model)
        {
            model.PlaceBox(start.X, start.Y, e.Position.X, e.Position.Y);
        }
    }
}
