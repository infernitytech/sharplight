using Avalonia.Controls;
using Avalonia.Interactivity;
using Sharplight.Studio.ViewModels.Inspector;

namespace Sharplight.Studio.Views.Inspector;

/// <summary>The inspector's sections tab: the file layout region by region (see <see cref="ViewModels.Inspector.SectionsViewModel"/>).</summary>
internal sealed partial class InspectorSectionsView : UserControl
{
    /// <summary>Creates the view.</summary>
    public InspectorSectionsView() => InitializeComponent();

    private void OnShow(object? sender, RoutedEventArgs e)
    {
        if (DataContext is SectionsViewModel sections)
        {
            sections.Show(sections.SelectedRow);
        }
    }
}
