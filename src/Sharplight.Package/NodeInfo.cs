namespace Sharplight.Package;

/// <summary>One node of a stored graph.</summary>
/// <param name="Id">Node id, unique within the graph.</param>
/// <param name="Op">Op name.</param>
/// <param name="Group">Placement group (e.g. the layer), empty when none.</param>
/// <param name="Attributes">Attributes formatted as <c>{key=value, ...}</c>.</param>
/// <param name="Inputs">Input values formatted as <c>%id:name type[shape]</c> (weights, states and constants included).</param>
/// <param name="Outputs">Output values.</param>
public sealed record NodeInfo(int Id, string Op, string Group, string Attributes, IReadOnlyList<string> Inputs, IReadOnlyList<GraphValueInfo> Outputs)
{
    /// <summary>The inputs in order, with their kind and producing node: the dataflow edges into this node.</summary>
    public IReadOnlyList<NodeInputInfo> InputValues { get; init; } = [];
}
