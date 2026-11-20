using Sharplight.Graph;

namespace Sharplight.Shaders.Lowering;

/// <summary>
/// Sibling projections that share their activations (e.g. q / k / v): weight matmuls of one format and K, lowered each on its own
/// and, at the row counts where every one of them runs a fused split-K GEMM, also as one multi-output GEMM
/// (<see cref="GraphLowerer"/>, splitk.slang MULTI) that replaces their split launches.
/// </summary>
internal sealed class MultiGemmGroup
{
    /// <summary>The matmul nodes in the order of the concatenated weight rows (every width but the last a multiple of the column tile).</summary>
    public required List<Node> Members { get; init; }

    /// <summary>The split launches of the members, captured while they are lowered (in member order once complete).</summary>
    public Dictionary<Node, MultiGemmPart> Parts { get; } = [];

    /// <summary>Members lowered so far; the multi-output GEMM is emitted after the last.</summary>
    public int Lowered { get; set; }

    /// <summary>False once a member's split launches turned out not to fit the multi-output GEMM.</summary>
    public bool Usable { get; set; } = true;

    /// <summary>The bindings the multi-output GEMM serves (null until it is emitted); the members' split launches skip them.</summary>
    public Func<Bindings, bool>? Serves { get; set; }
}
