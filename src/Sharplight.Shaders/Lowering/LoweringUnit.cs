using Sharplight.Graph;

namespace Sharplight.Shaders.Lowering;

/// <summary>
/// One scheduling unit of <see cref="GraphLowerer"/>: a single node, or a fused elementwise group together with the
/// nodes that are computed inside its kernel. Units keep graph order until the scheduler reorders them by level.
/// </summary>
internal sealed class LoweringUnit
{
    /// <summary>The node of a single-node unit; null for fused groups.</summary>
    public Node? Node;

    /// <summary>The elementwise nodes of a fused group; null for single-node units.</summary>
    public List<Node>? Fused;

    /// <summary>Values of a fused group that are consumed outside the group or are segment outputs.</summary>
    public List<Value>? FusedOutputs;

    /// <summary>Row normalization (RmsNorm/L2Norm) of a group output, computed in the same kernel.</summary>
    public Node? NormTail;

    /// <summary>Row normalization whose output feeds this group, computed in the same kernel (e.g. gated RmsNorm).</summary>
    public Node? NormHead;

    /// <summary>Two matmuls (sharing x) whose outputs only feed this group: lowered as one dual matvec for small batches.</summary>
    public (Node A, Node B)? Dual;

    /// <summary>A convolution whose output only feeds this group: one launch computes it with the group as its store epilogue.</summary>
    public Node? ConvHead;

    /// <summary>
    /// An Add of a constant row vector (a bias) that is the only consumer of this single-node unit's MatMul: single activation
    /// rows add it in the matvec's store instead of a separate elementwise launch.
    /// </summary>
    public Node? BiasAdd;

    /// <summary>
    /// GatherAxis nodes (rows of a table selected by indices, e.g. the per-expert biases of the selected experts) whose
    /// rows this fused group reads in place instead of from a materialized gather.
    /// </summary>
    public List<Node>? Gathers;

    /// <summary>Nodes computed inside this unit's kernel (e.g. L2Norm of delta-rule q/k).</summary>
    public List<Node>? Absorbed;

    /// <summary>Dependency level (ASAP): launches of one level are mutually independent.</summary>
    public int Level;

    /// <summary>True when the unit only re-interprets memory (no launch).</summary>
    public bool IsView => Node is { Op.Category: OpCategory.View };

    /// <summary>All graph nodes this unit implements, in execution order.</summary>
    public IEnumerable<Node> Nodes
    {
        get
        {
            if (Fused is null)
            {
                return BiasAdd is null ? [.. Absorbed ?? [], Node!] : [.. Absorbed ?? [], Node!, BiasAdd];
            }

            Node[] dual = Dual is { } pair ? [pair.A, pair.B] : ConvHead is null ? [] : [ConvHead];
            Node[] head = NormHead is null ? [] : [NormHead];
            Node[] tail = NormTail is null ? [] : [NormTail];
            return [.. Gathers ?? [], .. dual, .. head, .. Fused, .. tail];
        }
    }
}
