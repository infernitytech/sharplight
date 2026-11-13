using Sharplight.Tensors;

namespace Sharplight.Graph.Optimization;

/// <summary>
/// Bypasses nodes that return their operand unchanged (same type, shape and values): reshapes to the same shape, identity
/// permutations, full slices, broadcasts and tiles that change nothing, zero padding, f32→f32 and i32→i32 casts, and
/// <c>Contiguous</c> of a value that is already dense (a graph input, a constant, or the result of a computing op). The bypassed
/// node becomes dead and is removed by <see cref="DeadNodeElimination"/>. Graph outputs are never bypassed, and a weight or state
/// operand never takes the place of a computed value.
/// </summary>
public sealed class IdentityElimination : IGraphPass
{
    /// <inheritdoc/>
    public string Name => "identities";

    /// <inheritdoc/>
    public bool RepeatUntilStable => true;

    /// <inheritdoc/>
    public ComputeGraph Run(ComputeGraph graph)
    {
        var rewriter = new GraphRewriter(graph);
        foreach (var n in graph.Nodes)
        {
            if (n.Outputs.Length != 1 || n.Inputs.Length != 1 || GraphFacts.IsPinned(n, graph))
            {
                continue;
            }

            var x = rewriter.Resolve(n.Inputs[0]);
            if (rewriter.CanSubstitute(n.Output, x) && IsIdentity(n, x, rewriter))
            {
                rewriter.Substitute(n.Output, x);
            }
        }

        return rewriter.Build();
    }

    private static bool IsIdentity(Node n, Value x, GraphRewriter rewriter)
    {
        var a = n.Attributes;
        return n.Op.Name switch
        {
            "Reshape" or "Expand" => true,
            "Permute" => a.Get<int[]>("perm").Select((p, k) => p == k).All(same => same),
            "Slice" => a.Get<long>("start") == 0,
            "Tile" => a.Get<long[]>("repeats").All(r => r == 1),
            "Pad" => a.Get<long[]>("pads").All(p => p == 0),
            "Cast" => x.Type == DataType.F32 || x.Type == DataType.I32,
            "Contiguous" => IsDense(x, rewriter),
            _ => false,
        };
    }

    /// <summary>Whether the value is certainly a dense row-major buffer of its own (not a strided view of another value).</summary>
    private static bool IsDense(Value x, GraphRewriter rewriter)
    {
        if (x.Kind is ValueKind.Input or ValueKind.Constant)
        {
            return true;
        }

        return rewriter.ProducerOf(x) is { } producer
               && (producer.Op.Category is OpCategory.Elementwise or OpCategory.MatMul or OpCategory.Reduction or OpCategory.Composite
                   || producer.Op.Name is "Contiguous" or "Concat");
    }
}
