using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Platform.Storage;
using Sharplight.Studio.ViewModels.Audio;

namespace Sharplight.Studio.Views;

/// <summary>The Audio page. Code-behind takes audio files dropped on the input card.</summary>
internal sealed partial class AudioPage : UserControl
{
    public AudioPage()
    {
        InitializeComponent();
        InputCard.AddHandler(DragDrop.DragOverEvent, (_, e) => e.DragEffects = e.DataTransfer.Contains(DataFormat.File) ? DragDropEffects.Copy : DragDropEffects.None);
        InputCard.AddHandler(DragDrop.DropEvent, OnDrop);
    }

    private async void OnDrop(object? sender, DragEventArgs e)
    {
        if (DataContext is AudioPageViewModel page && e.DataTransfer.TryGetFiles() is { Length: > 0 } files && files[0].TryGetLocalPath() is { } path)
        {
            await page.SetAudioAsync(path);
        }
    }
}
