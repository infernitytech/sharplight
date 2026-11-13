using Avalonia.Controls;

namespace Sharplight.PackageViewer.Views;

/// <summary>The graphs tab (see <see cref="ViewModels.GraphsViewModel"/>).</summary>
public sealed partial class GraphsView : UserControl
{
    public GraphsView() => InitializeComponent();

    /// <summary>Keeps a node selected from the IR view visible (the data grid does not scroll to a programmatic selection).</summary>
    private void OnNodeSelectionChanged(object? sender, SelectionChangedEventArgs e)
    {
        if (sender is DataGrid grid && grid.SelectedItem is { } item)
        {
            grid.ScrollIntoView(item, null);
        }
    }
}
