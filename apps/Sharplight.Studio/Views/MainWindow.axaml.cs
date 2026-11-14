using Avalonia.Controls;
using Sharplight.Studio.ViewModels;

namespace Sharplight.Studio.Views;

/// <summary>The main window: navigation, page host and status bar.</summary>
internal sealed partial class MainWindow : Window
{
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
    }
}
