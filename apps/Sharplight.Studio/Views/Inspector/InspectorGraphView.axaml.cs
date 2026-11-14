using Avalonia.Controls;
using Avalonia.Interactivity;
using Sharplight.Studio.ViewModels.Inspector;

namespace Sharplight.Studio.Views.Inspector;

/// <summary>
/// One graph: inputs, outputs and ops on the left; the visual graph, the IR or the node list in the middle; the selected node's
/// details on the right (see <see cref="ViewModels.Inspector.GraphViewModel"/>).
/// </summary>
internal sealed partial class InspectorGraphView : UserControl
{
    private const double ZoomStep = 1.25;

    /// <summary>Creates the view.</summary>
    public InspectorGraphView() => InitializeComponent();

    private void OnZoomIn(object? sender, RoutedEventArgs e) => Canvas.ZoomBy(ZoomStep);

    private void OnZoomOut(object? sender, RoutedEventArgs e) => Canvas.ZoomBy(1 / ZoomStep);

    private void OnFit(object? sender, RoutedEventArgs e) => Canvas.FitAll();

    private void OnTop(object? sender, RoutedEventArgs e) => Canvas.ResetView();

    private void OnCenter(object? sender, RoutedEventArgs e) => Canvas.CenterOnSelection();

    /// <summary>Selects the node whose id a producer or consumer button carries in its tag.</summary>
    private void OnGoToNode(object? sender, RoutedEventArgs e)
    {
        if (sender is Control { Tag: int id } && DataContext is GraphViewModel graph)
        {
            graph.GoToNode(id);
        }
    }

    /// <summary>Keeps a node selected from another view visible in the list.</summary>
    private void OnNodeSelectionChanged(object? sender, SelectionChangedEventArgs e)
    {
        if (sender is ListBox list && list.SelectedItem is { } item)
        {
            list.ScrollIntoView(item);
        }
    }
}
