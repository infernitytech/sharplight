using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Platform.Storage;
using Sharplight.Studio.ViewModels.Models;

namespace Sharplight.Studio.Views;

/// <summary>The local model library. Files and folders dropped on it are validated and imported.</summary>
internal sealed partial class LibraryView : UserControl
{
    public LibraryView()
    {
        InitializeComponent();
        DropArea.AddHandler(DragDrop.DragOverEvent, OnDragOver);
        DropArea.AddHandler(DragDrop.DragLeaveEvent, (_, _) => DropArea.Classes.Remove("drop"));
        DropArea.AddHandler(DragDrop.DropEvent, OnDrop);
    }

    private void OnDragOver(object? sender, DragEventArgs e)
    {
        bool files = e.DataTransfer.Contains(DataFormat.File);
        e.DragEffects = files ? DragDropEffects.Copy : DragDropEffects.None;
        if (files)
        {
            DropArea.Classes.Add("drop");
        }
    }

    private async void OnDrop(object? sender, DragEventArgs e)
    {
        DropArea.Classes.Remove("drop");
        if (DataContext is not LibraryViewModel library || e.DataTransfer.TryGetFiles() is not { } items)
        {
            return;
        }

        var paths = items.Select(item => item.TryGetLocalPath()).OfType<string>().ToList();
        e.Handled = true;
        await library.ImportPathsAsync(paths);
    }
}
