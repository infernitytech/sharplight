using Avalonia.Controls;

namespace Sharplight.Studio.Views.Inspector;

/// <summary>The inspector's graphs tab: the graph list and the selected graph (see <see cref="ViewModels.Inspector.GraphsViewModel"/>).</summary>
internal sealed partial class InspectorGraphsView : UserControl
{
    /// <summary>Creates the view.</summary>
    public InspectorGraphsView() => InitializeComponent();
}
