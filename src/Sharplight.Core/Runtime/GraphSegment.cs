using Sharplight.Graph;

namespace Sharplight.Runtime;

/// <summary>A contiguous run of graph nodes placed on one device.</summary>
public sealed class GraphSegment
{
    /// <summary>The graph the segment was cut from.</summary>
    public required ComputeGraph Graph { get; init; }
    /// <summary>Position of the segment in execution order.</summary>
    public required int Index { get; init; }
    /// <summary>The nodes of the segment, in execution order.</summary>
    public required IReadOnlyList<Node> Nodes { get; init; }
    /// <summary>Values consumed by this segment but produced outside of it (graph inputs or earlier segments).</summary>
    public required IReadOnlyList<Value> Inputs { get; init; }
    /// <summary>Values produced here and needed by later segments or returned as graph outputs.</summary>
    public required IReadOnlyList<Value> Outputs { get; init; }
    /// <summary>Index into <see cref="PlacementPlan.Slots"/> of the device that runs the segment.</summary>
    public required int DeviceSlot { get; init; }

    /// <summary>The distinct weights the segment's nodes use.</summary>
    public IEnumerable<Value> Weights => Nodes.SelectMany(n => n.Inputs).Where(v => v.Kind == ValueKind.Weight).Distinct();
    /// <summary>The distinct states the segment's nodes use.</summary>
    public IEnumerable<Value> States => Nodes.SelectMany(n => n.Inputs).Where(v => v.Kind == ValueKind.State).Distinct();
}
