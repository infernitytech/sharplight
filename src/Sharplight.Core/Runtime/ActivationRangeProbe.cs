using System.Collections.Immutable;
using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Runtime;

/// <summary>
/// Measures the range of the activations a graph feeds into matrix products (<see cref="OpCategory.MatMul"/>: matmuls, convolutions)
/// and attention, the values reduced precision rounds to 16 bits (<see cref="ActivationPrecision"/>). <see cref="Instrument"/> adds a
/// per-row maximum of |x| for each such value as an extra output; after a run of the instrumented graph, <see cref="Read"/> takes the
/// largest. Run it once with strict precision (every activation exact, none lost to an earlier overflow) on a small, representative
/// input: the activation magnitudes of a checkpoint that overflows f16 come mostly from its weights, and
/// <see cref="ActivationRange.HalfSafeLimit"/> leaves the margin for larger inputs and sizes. A non-finite value counts as an overflow.
/// </summary>
public static class ActivationRangeProbe
{
    /// <summary>The prefix of the outputs <see cref="Instrument"/> adds (followed by an index).</summary>
    public const string OutputPrefix = "activation_range.";

    /// <summary>
    /// A copy of <paramref name="graph"/> with one extra f32 output per probed value (a float input or intermediate of rank 1 or more
    /// read by a matrix product, convolution or attention): its per-row maximum of |x|, named <see cref="OutputPrefix"/> + index. The
    /// original graph is unchanged (the copy shares its nodes).
    /// </summary>
    /// <param name="graph">The graph to measure.</param>
    /// <returns>The instrumented graph.</returns>
    public static ComputeGraph Instrument(ComputeGraph graph)
    {
        ArgumentNullException.ThrowIfNull(graph);
        var probed = graph.Nodes.Where(IsProduct).SelectMany(n => n.Inputs).Where(IsActivation).Distinct().ToList();
        int nextValue = graph.Nodes.SelectMany(n => n.Inputs.Concat(n.Outputs)).Concat(graph.Inputs).Select(v => v.Id).DefaultIfEmpty(-1).Max() + 1;
        int nextNode = graph.Nodes.Select(n => n.Id).DefaultIfEmpty(-1).Max() + 1;
        var abs = OpRegistry.Get("Abs");
        var nodes = new List<Node>(graph.Nodes);
        var outputs = new Dictionary<string, Value>(graph.Outputs);
        for (int i = 0; i < probed.Count; i++)
        {
            var x = probed[i];
            string group = x.Producer?.Group ?? "";
            var magnitude = Append(nodes, abs, x, $"{OutputPrefix}{i}.abs", group, ref nextNode, ref nextValue);
            outputs[$"{OutputPrefix}{i}"] = Append(nodes, StandardOps.ReduceMax, magnitude, $"{OutputPrefix}{i}", group, ref nextNode, ref nextValue);
        }

        return new ComputeGraph(graph.Name, nodes, graph.Inputs, outputs, graph.RuntimeScalars, graph.DynamicSymbols);
    }

    /// <summary>
    /// The largest magnitude the probe outputs of a run of an <see cref="Instrument"/>ed graph hold, and which value holds it.
    /// </summary>
    /// <param name="compiled">The compiled instrumented graph, after <see cref="CompiledModel.Run"/>.</param>
    /// <param name="parameters">The parameters of that run (they resolve the outputs' symbolic sizes).</param>
    /// <returns>The range; <see cref="ActivationRange.MaxMagnitude"/> 0 when the graph has no probed value.</returns>
    public static ActivationRange Read(CompiledModel compiled, RunParameters parameters)
    {
        ArgumentNullException.ThrowIfNull(compiled);
        ArgumentNullException.ThrowIfNull(parameters);
        float largest = 0;
        string? location = null;
        foreach (var (name, value) in compiled.Graph.Outputs.Where(o => o.Key.StartsWith(OutputPrefix, StringComparison.Ordinal)))
        {
            long count = compiled.OutputShape(name, parameters).Aggregate(1L, (a, b) => a * b);
            var rows = new float[count];
            compiled.GetOutput<float>(name, rows);
            float max = rows.Aggregate(0f, (m, r) => float.IsFinite(r) ? Math.Max(m, r) : float.PositiveInfinity);
            if (location is null || max > largest)
            {
                largest = max;
                // value = ReduceMax(Abs(x)): name x.
                location = value.Producer?.Inputs[0].Producer?.Inputs[0].Name;
            }
        }

        return new ActivationRange(largest, location);
    }

    private static bool IsProduct(Node node) => node.Op.Category == OpCategory.MatMul || ReferenceEquals(node.Op, StandardOps.Attention);

    private static bool IsActivation(Value value)
        => value.Kind is ValueKind.Input or ValueKind.Intermediate && value.Rank > 0 && (value.Type == DataType.F32 || value.Type == DataType.F16 || value.Type == DataType.BF16);

    private static Value Append(List<Node> nodes, OpSchema op, Value input, string name, string group, ref int nextNode, ref int nextValue)
    {
        ImmutableArray<Value> inputs = [input];
        var type = op.Infer(inputs, Attributes.Empty)[0];
        var node = new Node(nextNode++, op, inputs, Attributes.Empty, group);
        var output = new Value(nextValue++, name, type.Type, type.Shape, ValueKind.Intermediate) { Producer = node };
        node.Outputs = [output];
        nodes.Add(node);
        return output;
    }
}
