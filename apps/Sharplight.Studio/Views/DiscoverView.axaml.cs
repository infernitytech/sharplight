using Avalonia.Controls;
using Avalonia.Input;
using Sharplight.Studio.ViewModels.Models;

namespace Sharplight.Studio.Views;

/// <summary>Hugging Face search; Enter in the search box searches.</summary>
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
}
