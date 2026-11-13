using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.Platform.Storage;
using Sharplight.PackageViewer.ViewModels;

namespace Sharplight.PackageViewer.Views;

/// <summary>Main window: open button, drag and drop of a package file, one tab per aspect of the package.</summary>
public sealed partial class MainWindow : Window
{
    private static readonly FilePickerFileType s_packageFiles = new("Sharplight packages") { Patterns = ["*.slm"] };

    public MainWindow()
    {
        InitializeComponent();
        AddHandler(DragDrop.DragOverEvent, OnDragOver);
        AddHandler(DragDrop.DropEvent, OnDrop);
    }

    private MainWindowViewModel? ViewModel => DataContext as MainWindowViewModel;

    private async void OnOpenClick(object? sender, RoutedEventArgs e)
    {
        var files = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "Open Sharplight package",
            AllowMultiple = false,
            FileTypeFilter = [s_packageFiles, FilePickerFileTypes.All],
        });
        await OpenAsync(files.FirstOrDefault());
    }

    private static void OnDragOver(object? sender, DragEventArgs e)
    {
        e.DragEffects = e.DataTransfer.Contains(DataFormat.File) ? DragDropEffects.Copy : DragDropEffects.None;
    }

    private async void OnDrop(object? sender, DragEventArgs e)
    {
        await OpenAsync(e.DataTransfer.TryGetFile() as IStorageFile);
    }

    private async Task OpenAsync(IStorageFile? file)
    {
        if (file?.TryGetLocalPath() is { } path && ViewModel is { } viewModel)
        {
            await viewModel.OpenAsync(path);
        }
    }
}
