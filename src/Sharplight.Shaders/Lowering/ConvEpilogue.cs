using Sharplight.Graph;

namespace Sharplight.Shaders.Lowering;

/// <summary>
/// An elementwise group applied in a convolution's store (see <see cref="LoweringUnit.ConvHead"/>): the group's nodes in
/// order and the one value it materializes, of the convolution output's shape.
/// </summary>
internal sealed record ConvEpilogue(List<Node> Group, Value Output);
