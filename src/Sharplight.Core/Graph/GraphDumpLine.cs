namespace Sharplight.Graph;

/// <summary>One line of a graph dump (<see cref="ComputeGraph.DumpLines"/>).</summary>
/// <param name="Depth">Nesting depth: 0 for the graph header, 1 for its members, deeper inside control-flow subgraphs.</param>
/// <param name="Kind">What the line describes.</param>
/// <param name="Text">The line without indentation.</param>
/// <param name="Graph">The graph the line belongs to (a subgraph for lines inside control flow).</param>
/// <param name="Node">The node of a <see cref="GraphDumpLineKind.Node"/> line, else null.</param>
/// <param name="Value">The declared value of an input, weight, state or constant line, or the value of an output line, else null.</param>
public sealed record GraphDumpLine(int Depth, GraphDumpLineKind Kind, string Text, ComputeGraph Graph, Node? Node, Value? Value);
