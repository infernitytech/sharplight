namespace Sharplight.Graph;

/// <summary>Produces the lines of <see cref="ComputeGraph.Dump"/>, recursing into control-flow subgraphs held in node attributes.</summary>
internal static class GraphDump
{
    public static IEnumerable<GraphDumpLine> Lines(ComputeGraph graph, int depth, string? label)
    {
        var header = label is null ? $"graph {graph.Name}" : $"{label}: graph {graph.Name}";
        yield return new GraphDumpLine(depth, GraphDumpLineKind.Graph, header, graph, null, null);
        int member = depth + 1;
        foreach (var input in graph.Inputs)
        {
            yield return new GraphDumpLine(member, GraphDumpLineKind.Input, $"input  {input} {input.Input}", graph, null, input);
        }

        foreach (var value in graph.Nodes.SelectMany(n => n.Inputs).Distinct().Where(v => v.Kind is ValueKind.Weight or ValueKind.State or ValueKind.Constant))
        {
            yield return Declaration(member, graph, value);
        }

        foreach (var node in graph.Nodes)
        {
            yield return new GraphDumpLine(member, GraphDumpLineKind.Node, node.ToString(), graph, node, null);
            foreach (var (key, attribute) in node.Attributes)
            {
                if (attribute is ComputeGraph subgraph)
                {
                    foreach (var line in Lines(subgraph, member + 1, key))
                    {
                        yield return line;
                    }
                }
            }
        }

        foreach (var (name, value) in graph.Outputs)
        {
            yield return new GraphDumpLine(member, GraphDumpLineKind.Output, $"output {name} = %{value.Id}", graph, null, value);
        }

        foreach (var (symbol, bound) in graph.DynamicSymbols)
        {
            yield return new GraphDumpLine(member, GraphDumpLineKind.DynamicSymbol, $"dynamic {symbol} <= {bound}", graph, null, null);
        }
    }

    private static GraphDumpLine Declaration(int depth, ComputeGraph graph, Value value) => value.Kind switch
    {
        ValueKind.Weight => new(depth, GraphDumpLineKind.Weight, $"weight {value}", graph, null, value),
        ValueKind.State => new(depth, GraphDumpLineKind.State, $"state  {value} ({value.State?.Kind})", graph, null, value),
        _ => new(depth, GraphDumpLineKind.Constant, $"const  {value}", graph, null, value),
    };
}
