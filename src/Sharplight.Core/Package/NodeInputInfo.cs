using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Package;

/// <summary>One input of a stored node, with the reference needed to follow the dataflow edge it comes from.</summary>
/// <param name="ValueId">Id of the value within the graph (<c>%id</c> in the IR).</param>
/// <param name="Kind">Where the value comes from: another node (<see cref="ValueKind.Intermediate"/>), a graph input, a weight, a state or a constant.</param>
/// <param name="Name">Value name (the weight, input or state name for those kinds).</param>
/// <param name="Type">Element type.</param>
/// <param name="Shape">Shape; may contain symbolic dimensions.</param>
/// <param name="Producer">Id of the node that produces the value (<see cref="NodeInfo.Id"/>), or -1 when no node produces it.</param>
public sealed record NodeInputInfo(int ValueId, ValueKind Kind, string Name, DataType Type, Shape Shape, int Producer);
