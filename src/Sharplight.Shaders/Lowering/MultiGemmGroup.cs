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

    /// <summary>
    /// The split launches of the members, captured while they are lowered, per kernel family (true: the fp8 GEMM, false: gemm_cm2;
    /// an fp8 matmul splits K on gemm_cm2 below the fp8 GEMM's rows and on the fp8 GEMM above).
    /// </summary>
    public Dictionary<(Node Member, bool Fp8), MultiGemmPart> Parts { get; } = [];

    /// <summary>Members lowered so far; the multi-output GEMM is emitted after the last.</summary>
    public int Lowered { get; set; }

    /// <summary>False once a member's split launches turned out not to fit the multi-output GEMM.</summary>
    public bool Usable { get; set; } = true;

    /// <summary>The bindings each family's multi-output GEMM serves (absent until it is emitted); the members' split launches skip them.</summary>
    public Dictionary<bool, Func<Bindings, bool>> Serves { get; } = [];

    /// <summary>
    /// Launches that read a member's product and must follow the multi-output GEMM that writes it (the bias add of a member whose
    /// epilogue runs as its own launch), emitted right after it; see <see cref="GraphLowerer"/>'s epilogue lowering.
    /// </summary>
    public List<Action> AfterEmit { get; } = [];

    /// <summary>The kernel families whose multi-output GEMM adds the members' bias rows in its store (no bias launch after it).</summary>
    public HashSet<bool> StoresBias { get; } = [];

    /// <summary>The members' launches on the hand-written f16 GEMM's prompt-sized tiles, captured while they are lowered.</summary>
    public Dictionary<Node, MultiGemmTallPart> TallParts { get; } = [];

    /// <summary>
    /// The bindings the multi-output form of the hand-written f16 GEMM's prompt-sized tiles serves (null until it is emitted, or when
    /// it is not); the members' own launches skip them.
    /// </summary>
    public Func<Bindings, bool>? TallServes { get; set; }
}
