namespace Sharplight.Studio.Services.Inspection;

/// <summary>A box of the graph view: a node, a graph input or a graph output, with its position in graph coordinates.</summary>
internal sealed class LayoutNode
{
    /// <summary>Index in <see cref="GraphLayout.Nodes"/>.</summary>
    public required int Index { get; init; }

    /// <summary>The node id (<see cref="Sharplight.Package.NodeInfo.Id"/>) of an op box, -1 for inputs and outputs.</summary>
    public required int NodeId { get; init; }

    /// <summary>What the box stands for.</summary>
    public required LayoutNodeKind Kind { get; init; }

    /// <summary>Main text: the op name, or the input or output name.</summary>
    public required string Title { get; init; }

    /// <summary>Second line: the (first) output type and shape.</summary>
    public required string Subtitle { get; init; }

    /// <summary>Placement group (layer) of an op, empty otherwise.</summary>
    public string Group { get; init; } = "";

    /// <summary>Op family, for the color stripe.</summary>
    public OpCategory Category { get; init; }

    /// <summary>Number of weights the op reads.</summary>
    public int WeightCount { get; init; }

    /// <summary>Layer (row) of the layered layout: the longest path from the inputs.</summary>
    public int Rank { get; set; }

    /// <summary>Left edge.</summary>
    public double X { get; set; }

    /// <summary>Top edge.</summary>
    public double Y { get; set; }

    /// <summary>Box width.</summary>
    public double Width { get; init; }

    /// <summary>Box height.</summary>
    public double Height { get; init; }

    /// <summary>Horizontal center.</summary>
    public double CenterX => X + (Width / 2);

    /// <summary>Vertical center.</summary>
    public double CenterY => Y + (Height / 2);
}
