using Avalonia.Controls;
using Sharplight.Studio.ViewModels.Inspector;

namespace Sharplight.Studio.Views.Inspector;

/// <summary>The IR text of a graph with search and value navigation (see <see cref="IrViewModel"/>).</summary>
internal sealed partial class InspectorIrView : UserControl
{
    /// <summary>Creates the view.</summary>
    public InspectorIrView() => InitializeComponent();

    /// <summary>Selecting a producer, consumer or neighbor jumps to its line.</summary>
    private void OnReferenceSelected(object? sender, SelectionChangedEventArgs e)
    {
        if (DataContext is IrViewModel viewModel && e.AddedItems is [IrReferenceRow reference, ..])
        {
            viewModel.GoTo(reference.Target);
        }
    }

    /// <summary>Keeps the selected line visible (search, navigation and selection from the graph).</summary>
    private void OnLineSelected(object? sender, SelectionChangedEventArgs e)
    {
        if (sender is ListBox list && list.SelectedItem is { } item)
        {
            list.ScrollIntoView(item);
        }
    }
}
