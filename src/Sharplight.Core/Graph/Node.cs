using System.Collections.Immutable;

namespace Sharplight.Graph;

/// <summary>An operation in a <see cref="ComputeGraph"/>.</summary>
public sealed class Node
{
    public int Id { get; }
    public OpSchema Op { get; }
    public ImmutableArray<Value> Inputs { get; }
    public ImmutableArray<Value> Outputs { get; internal set; }
    public Attributes Attributes { get; }

    /// <summary>
    /// Placement group (e.g. <c>"layers.12"</c>). Nodes in the same group are kept on the same device by
    /// the partitioner; groups are assigned to devices in graph order (pipeline split).
    /// </summary>
    public string Group { get; }

    /// <summary>States written by this node (side effects). Execution order of nodes is preserved.</summary>
    public ImmutableArray<Value> Writes { get; init; } = [];

    internal Node(int id, OpSchema op, ImmutableArray<Value> inputs, Attributes attributes, string group)
    {
        Id = id;
        Op = op;
        Inputs = inputs;
        Attributes = attributes;
        Group = group;
    }

    public Value Output => Outputs.Length == 1 ? Outputs[0] : throw new InvalidOperationException($"{Op.Name} has {Outputs.Length} outputs.");

    public override string ToString()
    {
        var outs = string.Join(", ", Outputs);
        var ins = string.Join(", ", Inputs.Select(i => $"%{i.Id}"));
        var attrs = Attributes.Count > 0 ? " " + Attributes : "";
        return $"{outs} = {Op.Name}({ins}){attrs}  @{Group}";
    }
}
