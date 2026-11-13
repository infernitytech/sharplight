using Avalonia.Controls;
using Sharplight.PackageViewer.ViewModels;

namespace Sharplight.PackageViewer.Views;

/// <summary>The IR text of a graph with search and value navigation (see <see cref="IrViewModel"/>).</summary>
public sealed partial class IrView : UserControl
{
    public IrView() => InitializeComponent();

    /// <summary>Selecting a producer, consumer or neighbor jumps to its line.</summary>
    private void OnReferenceSelected(object? sender, SelectionChangedEventArgs e)
    {
        if (DataContext is IrViewModel viewModel && e.AddedItems is [IrReferenceRow reference, ..])
        {
            viewModel.GoTo(reference.Target);
        }
    }
}
