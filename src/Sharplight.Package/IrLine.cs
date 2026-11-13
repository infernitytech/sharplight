using Sharplight.Graph;

namespace Sharplight.Package;

/// <summary>
/// One line of a stored graph's textual IR (<see cref="GraphInfo.Ir"/>, the format of <see cref="ComputeGraph.Dump"/>), with the value
/// references needed to navigate it. Value ids are unique within a <see cref="Scope"/> only: every control-flow subgraph has its own.
/// </summary>
/// <param name="Depth">Nesting depth: 0 for the graph header, 1 for its members, deeper inside control-flow subgraphs.</param>
/// <param name="Kind">What the line describes.</param>
/// <param name="Text">The line without indentation.</param>
/// <param name="Scope">The graph the line belongs to: 0 for the stored graph, 1, 2, ... for its subgraphs in listing order.</param>
/// <param name="NodeId">Node id within the scope for <see cref="GraphDumpLineKind.Node"/> lines (the <see cref="NodeInfo.Id"/> in scope 0), else -1.</param>
/// <param name="Defines">Ids of the values the line declares (inputs, weights, states, constants) or produces (node outputs).</param>
/// <param name="Uses">Ids of the values the line reads (node inputs, the value of an output line).</param>
public sealed record IrLine(int Depth, GraphDumpLineKind Kind, string Text, int Scope, int NodeId, IReadOnlyList<int> Defines, IReadOnlyList<int> Uses)
{
    internal static IReadOnlyList<IrLine> Build(ComputeGraph graph)
    {
        var scopes = new Dictionary<ComputeGraph, int>(ReferenceEqualityComparer.Instance);
        var lines = new List<IrLine>();
        foreach (var line in graph.DumpLines())
        {
            if (!scopes.TryGetValue(line.Graph, out int scope))
            {
                scope = scopes.Count;
                scopes.Add(line.Graph, scope);
            }

            lines.Add(From(line, scope));
        }

        return lines;
    }

    private static IrLine From(GraphDumpLine line, int scope)
    {
        if (line.Node is { } node)
        {
            int[] defines = [.. node.Outputs.Select(v => v.Id)];
            int[] uses = [.. node.Inputs.Select(v => v.Id).Distinct()];
            return new IrLine(line.Depth, line.Kind, line.Text, scope, node.Id, defines, uses);
        }

        int[] value = line.Value is { } v ? [v.Id] : [];
        bool reads = line.Kind == GraphDumpLineKind.Output;
        return new IrLine(line.Depth, line.Kind, line.Text, scope, -1, reads ? [] : value, reads ? value : []);
    }
}
