using CommunityToolkit.Mvvm.ComponentModel;
using Sharplight.Package;

namespace Sharplight.Studio.ViewModels.Inspector;

/// <summary>The graphs tab: the stored graphs and the selected one.</summary>
internal sealed partial class GraphsViewModel : ObservableObject
{
    /// <param name="graphs">The stored graphs.</param>
    /// <param name="weights">The package's weight index by name.</param>
    public GraphsViewModel(IReadOnlyList<GraphInfo> graphs, IReadOnlyDictionary<string, PackageWeightInfo> weights)
    {
        Graphs = [.. graphs.Select(g => new GraphViewModel(g, weights))];
    }

    /// <summary>Every stored graph, by name.</summary>
    public IReadOnlyList<GraphViewModel> Graphs { get; }

    /// <summary>True when the definition holds no readable graph.</summary>
    public bool IsEmpty => Graphs.Count == 0;

    /// <summary>The graph shown; its nodes, IR and layout load when it is selected.</summary>
    [ObservableProperty]
    public partial GraphViewModel? SelectedGraph { get; set; }

    partial void OnSelectedGraphChanged(GraphViewModel? value) => _ = value?.LoadAsync();
}
