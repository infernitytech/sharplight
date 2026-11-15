using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Platform.Storage;
using Sharplight.Studio.ViewModels.Inference;

namespace Sharplight.Studio.Views.Inference;

/// <summary>The image input block of an inference page. Code-behind takes the image files dropped on it.</summary>
internal sealed partial class ImageInputView : UserControl
{
    public ImageInputView()
    {
        InitializeComponent();
        DropArea.AddHandler(DragDrop.DragOverEvent, (_, e) => e.DragEffects = e.DataTransfer.Contains(DataFormat.File) ? DragDropEffects.Copy : DragDropEffects.None);
        DropArea.AddHandler(DragDrop.DropEvent, OnDrop);
    }

    private async void OnDrop(object? sender, DragEventArgs e)
    {
        if (DataContext is ImageInputViewModel input && e.DataTransfer.TryGetFiles() is { Length: > 0 } files && files[0].TryGetLocalPath() is { } path)
        {
            await input.SetFileAsync(path);
        }
    }
}
