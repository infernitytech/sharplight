using CommunityToolkit.Mvvm.ComponentModel;
using Sharplight.Package;

namespace Sharplight.PackageViewer.ViewModels;

/// <summary>The graphs tab: the list of stored graphs and the selected one.</summary>
public sealed partial class GraphsViewModel : ObservableObject
{
    public GraphsViewModel(IReadOnlyList<GraphInfo> graphs)
    {
        Graphs = [.. graphs.Select(g => new GraphViewModel(g))];
    }

    public IReadOnlyList<GraphViewModel> Graphs { get; }

    /// <summary>The graph shown on the right; its node table loads when it is selected.</summary>
    [ObservableProperty]
    public partial GraphViewModel? SelectedGraph { get; set; }

    partial void OnSelectedGraphChanged(GraphViewModel? value) => _ = value?.LoadAsync();
}
