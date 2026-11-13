using System.Collections.Immutable;

namespace Sharplight.Graph;

/// <summary>An operation in a <see cref="ComputeGraph"/>.</summary>
public sealed class Node
{
    /// <summary>Unique id of the node within its graph, assigned in emission order.</summary>
    public int Id { get; }
    /// <summary>The operator this node applies.</summary>
    public OpSchema Op { get; }
    /// <summary>The operands, in the order the op schema defines.</summary>
    public ImmutableArray<Value> Inputs { get; }
    /// <summary>The values this node produces; their types and shapes come from the op's shape inference.</summary>
    public ImmutableArray<Value> Outputs { get; internal set; }
    /// <summary>Static parameters of the op (axes, epsilons, names of runtime scalars, ...).</summary>
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

    /// <summary>
    /// The single output of a one-output node.
    /// </summary>
    /// <exception cref="InvalidOperationException">The node has zero or several outputs.</exception>
    public Value Output => Outputs.Length == 1 ? Outputs[0] : throw new InvalidOperationException($"{Op.Name} has {Outputs.Length} outputs.");

    public override string ToString()
    {
        var outs = string.Join(", ", Outputs);
        var ins = string.Join(", ", Inputs.Select(i => $"%{i.Id}"));
        var attrs = Attributes.Count > 0 ? " " + Attributes : "";
        return $"{outs} = {Op.Name}({ins}){attrs}  @{Group}";
    }
}
