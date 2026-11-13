using Sharplight.Graph;
using Sharplight.Models;

namespace Sharplight.Inspect;

/// <summary>Writes a summary of a model definition and an abbreviated dump (first and last lines) of one of its compute graphs.</summary>
internal sealed class GraphReport
{
    private const int HeadLines = 80;
    private const int TailLines = 8;

    private readonly ModelDefinition _definition;
    private readonly ComputeGraph _graph;
    private readonly TextWriter _output;

    public GraphReport(ModelDefinition definition, ComputeGraph graph, TextWriter output)
    {
        _definition = definition;
        _graph = graph;
        _output = output;
    }

    public void Write()
    {
        _output.WriteLine($"{_definition.Name}: {_definition.Weights.Count} weights, {_definition.States.Count} states, graphs: {string.Join(", ", _definition.Graphs.Keys)}");
        _output.WriteLine($"{_graph.Nodes.Count} nodes; ops: {FormatOpCounts()}");
        WriteAbbreviatedDump();
    }

    private string FormatOpCounts()
    {
        var counts = _graph.Nodes
            .GroupBy(node => node.Op.Name)
            .OrderByDescending(group => group.Count())
            .Select(group => $"{group.Key}×{group.Count()}");
        return string.Join(", ", counts);
    }

    private void WriteAbbreviatedDump()
    {
        var lines = _graph.Dump().Split('\n');
        foreach (var line in lines.Take(HeadLines))
        {
            _output.WriteLine(line);
        }

        _output.WriteLine("...");
        foreach (var line in lines.TakeLast(TailLines))
        {
            _output.WriteLine(line);
        }
    }
}
