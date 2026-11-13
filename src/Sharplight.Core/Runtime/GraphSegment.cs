using Sharplight.Graph;

namespace Sharplight.Runtime;

/// <summary>A contiguous run of graph nodes placed on one device.</summary>
public sealed class GraphSegment
{
    public required ComputeGraph Graph { get; init; }
    public required int Index { get; init; }
    public required IReadOnlyList<Node> Nodes { get; init; }
    /// <summary>Values consumed by this segment but produced outside of it (graph inputs or earlier segments).</summary>
    public required IReadOnlyList<Value> Inputs { get; init; }
    /// <summary>Values produced here and needed by later segments or returned as graph outputs.</summary>
    public required IReadOnlyList<Value> Outputs { get; init; }
    public required int DeviceSlot { get; init; }

    public IEnumerable<Value> Weights => Nodes.SelectMany(n => n.Inputs).Where(v => v.Kind == ValueKind.Weight).Distinct();
    public IEnumerable<Value> States => Nodes.SelectMany(n => n.Inputs).Where(v => v.Kind == ValueKind.State).Distinct();
}
