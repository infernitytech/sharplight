using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using Sharplight.Studio.ViewModels.Models;

namespace Sharplight.Studio.Views;

/// <summary>Hugging Face search; Enter in the search box searches, a highlight card opens its repository.</summary>
internal sealed partial class DiscoverView : UserControl
{
    public DiscoverView()
    {
        InitializeComponent();
        QueryBox.KeyDown += (_, e) =>
        {
            if (e.Key == Key.Enter && DataContext is HubSearchViewModel viewModel)
            {
                e.Handled = true;
                viewModel.SearchCommand.Execute(null);
            }
        };
    }

    private void OnHighlightClick(object? sender, RoutedEventArgs e)
    {
        if (DataContext is HubSearchViewModel viewModel && sender is Control { DataContext: HubResultViewModel result })
        {
            viewModel.SelectResultCommand.Execute(result);
        }
    }
}
