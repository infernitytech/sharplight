using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Platform.Storage;
using Sharplight.Studio.ViewModels.TimeSeries;

namespace Sharplight.Studio.Views;

/// <summary>The Time series page. Code-behind takes the table file dropped on the input card.</summary>
internal sealed partial class TimeSeriesPage : UserControl
{
    public TimeSeriesPage()
    {
        InitializeComponent();
        InputCard.AddHandler(DragDrop.DragOverEvent, (_, e) => e.DragEffects = e.DataTransfer.Contains(DataFormat.File) ? DragDropEffects.Copy : DragDropEffects.None);
        InputCard.AddHandler(DragDrop.DropEvent, OnDrop);
    }

    private async void OnDrop(object? sender, DragEventArgs e)
    {
        if (DataContext is TimeSeriesPageViewModel page && e.DataTransfer.TryGetFiles() is { Length: > 0 } files && files[0].TryGetLocalPath() is { } path)
        {
            await page.SetTableAsync(path);
        }
    }
}
