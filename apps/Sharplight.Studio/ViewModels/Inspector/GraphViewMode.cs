namespace Sharplight.Studio.ViewModels.Inspector;

/// <summary>How the selected graph is shown.</summary>
internal enum GraphViewMode
{
    /// <summary>The visual dataflow graph.</summary>
    Graph,

    /// <summary>The textual IR.</summary>
    Ir,

    /// <summary>The filterable node list.</summary>
    Nodes,
}
