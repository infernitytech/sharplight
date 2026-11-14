namespace Sharplight.Studio.Services.Inspection;

/// <summary>What a box of the graph view stands for.</summary>
internal enum LayoutNodeKind
{
    /// <summary>A node (op) of the graph.</summary>
    Op,

    /// <summary>A graph input.</summary>
    Input,

    /// <summary>A graph output.</summary>
    Output,
}
