using System.Collections.Immutable;
using Sharplight.Tensors;

namespace Sharplight.Graph.Optimization;

/// <summary>
/// Collects edits to a <see cref="ComputeGraph"/> (value substitutions, removed nodes, re-emitted nodes, new constants) and
/// builds the edited graph. The source graph is never modified: intermediates and nodes are copied (values keep their ids and
/// names, nodes are renumbered by position), while leaves (inputs, weights, states, constants) are shared, so callers holding input values keep using them.
/// </summary>
internal sealed class GraphRewriter
{
    private readonly Dictionary<Value, Value> _substitutions = [];
    private readonly HashSet<Node> _removed = [];
    private readonly Dictionary<Node, (OpSchema Op, Attributes Attributes, ImmutableArray<Value> Inputs)> _rewritten = [];
    private readonly HashSet<Value> _graphOutputs;
    private int _nextValueId;

    public GraphRewriter(ComputeGraph graph)
    {
        Graph = graph;
        _graphOutputs = [.. graph.Outputs.Values];
        _nextValueId = graph.Inputs.Concat(graph.Outputs.Values)
            .Concat(graph.Nodes.SelectMany(n => n.Inputs.Concat(n.Outputs).Concat(n.Writes)))
            .Select(v => v.Id).DefaultIfEmpty(-1).Max() + 1;
    }

    /// <summary>The graph being edited.</summary>
    public ComputeGraph Graph { get; }

    /// <summary>Whether any edit was recorded.</summary>
    public bool HasChanges => _substitutions.Count > 0 || _removed.Count > 0 || _rewritten.Count > 0;

    /// <summary>Whether the value is one of the graph's named outputs.</summary>
    public bool IsGraphOutput(Value v) => _graphOutputs.Contains(v);

    /// <summary>The value that replaces <paramref name="v"/> after the substitutions recorded so far.</summary>
    public Value Resolve(Value v)
    {
        while (_substitutions.TryGetValue(v, out var next))
        {
            v = next;
        }

        return v;
    }

    /// <summary>The inputs of a node after substitutions (and after its own rewrite, if any).</summary>
    public ImmutableArray<Value> InputsOf(Node n)
    {
        var inputs = _rewritten.TryGetValue(n, out var r) ? r.Inputs : n.Inputs;
        return [.. inputs.Select(Resolve)];
    }

    /// <summary>The op and attributes of a node after its rewrite, if any.</summary>
    public (OpSchema Op, Attributes Attributes) OpOf(Node n) => _rewritten.TryGetValue(n, out var r) ? (r.Op, r.Attributes) : (n.Op, n.Attributes);

    /// <summary>The node that produces <paramref name="v"/> in the edited graph, or <see langword="null"/> for leaves and removed nodes.</summary>
    public Node? ProducerOf(Value v)
    {
        var resolved = Resolve(v);
        return resolved.Producer is { } p && !_removed.Contains(p) ? p : null;
    }

    /// <summary>
    /// Whether <paramref name="old"/> may be replaced by <paramref name="replacement"/>: same type and shape, <paramref name="old"/> is
    /// an intermediate or constant that is not a graph output, and the replacement is not a weight or state (backends treat
    /// those operands specially, so they never take the place of a computed value).
    /// </summary>
    public bool CanSubstitute(Value old, Value replacement)
        => !ReferenceEquals(old, replacement)
           && old.Kind is ValueKind.Intermediate or ValueKind.Constant
           && replacement.Kind is ValueKind.Intermediate or ValueKind.Input or ValueKind.Constant
           && !IsGraphOutput(old)
           && old.Type == replacement.Type && old.Shape.Equals(replacement.Shape);

    /// <summary>Makes every use of <paramref name="old"/> use <paramref name="replacement"/> (which must be available before those uses).</summary>
    /// <exception cref="InvalidOperationException"><see cref="CanSubstitute"/> is false.</exception>
    public void Substitute(Value old, Value replacement)
    {
        replacement = Resolve(replacement);
        if (!CanSubstitute(old, replacement))
        {
            throw new InvalidOperationException($"Cannot substitute {old} by {replacement}.");
        }

        _substitutions[old] = replacement;
    }

    /// <summary>Drops a node; none of its outputs may still be used.</summary>
    public void Remove(Node n) => _removed.Add(n);

    /// <summary>Whether the node was dropped.</summary>
    public bool IsRemoved(Node n) => _removed.Contains(n);

    /// <summary>
    /// Re-emits <paramref name="n"/> as <paramref name="op"/> over <paramref name="inputs"/> with <paramref name="attributes"/>, keeping
    /// its outputs. The rewrite is only recorded when the op's shape inference yields exactly the node's output types and shapes.
    /// </summary>
    /// <returns>Whether the rewrite was recorded.</returns>
    public bool TryRewrite(Node n, OpSchema op, Attributes attributes, IReadOnlyList<Value> inputs)
    {
        if (n.Writes.Length > 0)
        {
            return false;
        }

        var resolved = inputs.Select(Resolve).ToImmutableArray();
        IReadOnlyList<TensorType> types;
        try
        {
            types = op.Infer(resolved, attributes);
        }
        catch (Exception e) when (e is ArgumentException or InvalidOperationException or NotSupportedException or InvalidCastException or KeyNotFoundException or IndexOutOfRangeException)
        {
            return false;
        }

        if (types.Count != n.Outputs.Length)
        {
            return false;
        }

        for (int k = 0; k < types.Count; k++)
        {
            if (types[k].Type != n.Outputs[k].Type || !types[k].Shape.Equals(n.Outputs[k].Shape))
            {
                return false;
            }
        }

        _rewritten[n] = (op, attributes, resolved);
        return true;
    }

    /// <summary>Creates a constant value (a leaf shared by the edited graph).</summary>
    public Value NewConstant(string name, DataType type, Shape shape, byte[] data)
        => new(_nextValueId++, name, type, shape, ValueKind.Constant) { ConstantData = data };

    /// <summary>
    /// Builds the edited graph, or returns the source graph when nothing changed.
    /// </summary>
    /// <exception cref="InvalidOperationException">An edit left a use of a value that is no longer produced before it.</exception>
    public ComputeGraph Build()
    {
        if (!HasChanges)
        {
            return Graph;
        }

        var copies = new Dictionary<Value, Value>();
        Value Map(Value v)
        {
            v = Resolve(v);
            if (v.Kind != ValueKind.Intermediate)
            {
                return v;
            }

            return copies.TryGetValue(v, out var copy)
                ? copy
                : throw new InvalidOperationException($"Graph '{Graph.Name}': value {v} is used but no longer produced before the use.");
        }

        var nodes = new List<Node>(Graph.Nodes.Count);
        foreach (var n in Graph.Nodes)
        {
            if (_removed.Contains(n))
            {
                continue;
            }

            var (op, attributes, inputs) = _rewritten.TryGetValue(n, out var r) ? r : (n.Op, n.Attributes, n.Inputs);
            // Node ids stay equal to the position in the node list, like the builder assigns them (the planner relies on it for speed).
            var node = new Node(nodes.Count, op, [.. inputs.Select(Map)], attributes, n.Group) { Writes = [.. n.Writes.Select(Map)] };
            var outputs = ImmutableArray.CreateBuilder<Value>(n.Outputs.Length);
            foreach (var o in n.Outputs)
            {
                var copy = new Value(o.Id, o.Name, o.Type, o.Shape, ValueKind.Intermediate) { Producer = node };
                copies[o] = copy;
                outputs.Add(copy);
            }

            node.Outputs = outputs.MoveToImmutable();
            nodes.Add(node);
        }

        var graphOutputs = new Dictionary<string, Value>(Graph.Outputs.Count);
        foreach (var (name, v) in Graph.Outputs)
        {
            graphOutputs[name] = Map(v);
        }

        return new ComputeGraph(Graph.Name, nodes, Graph.Inputs, graphOutputs, Graph.RuntimeScalars, Graph.DynamicSymbols);
    }
}
