using Sharplight.Graph;

namespace Sharplight.Shaders.Lowering;

/// <summary>
/// An elementwise group applied in a weight matmul's store (see <see cref="LoweringUnit.MatMulHead"/>): the group's nodes in
/// order and the one value it materializes, of the matmul output's shape.
/// </summary>
internal sealed record MatMulEpilogue(List<Node> Group, Value Output);
