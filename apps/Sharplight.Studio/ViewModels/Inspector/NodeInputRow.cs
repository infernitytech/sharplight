namespace Sharplight.Studio.ViewModels.Inspector;

/// <summary>An input of the selected node.</summary>
/// <param name="Kind">Where it comes from: <c>node</c>, <c>input</c>, <c>weight</c>, <c>state</c> or <c>constant</c>.</param>
/// <param name="Name">Value name.</param>
/// <param name="TypeShape">Type and shape, e.g. <c>q4_K[1024, 3072]</c>.</param>
/// <param name="Producer">Id of the producing node, or -1.</param>
/// <param name="Detail">Extra text: the producing op, or the stored size of a weight.</param>
internal sealed record NodeInputRow(string Kind, string Name, string TypeShape, int Producer, string Detail)
{
    /// <summary>Whether a node produces the value (so the row links to it).</summary>
    public bool HasProducer => Producer >= 0;
}
