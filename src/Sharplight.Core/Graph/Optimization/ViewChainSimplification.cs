namespace Sharplight.Graph.Optimization;

/// <summary>
/// Collapses chains of views: <c>Reshape(Reshape(x))</c> becomes one reshape of <c>x</c>, <c>Permute(Permute(x))</c> one permutation
/// (an identity permutation is then removed by <see cref="IdentityElimination"/>), and <c>Slice(Slice(x))</c> along the same axis
/// one slice. A rewrite is applied only when the op's shape inference over the new operand reproduces the node's output type
/// and shape exactly, including symbolic dimensions; the inner view stays while other nodes use it.
/// </summary>
public sealed class ViewChainSimplification : IGraphPass
{
    /// <inheritdoc/>
    public string Name => "view-chains";

    /// <inheritdoc/>
    public bool RepeatUntilStable => true;

    /// <inheritdoc/>
    public ComputeGraph Run(ComputeGraph graph)
    {
        var rewriter = new GraphRewriter(graph);
        foreach (var n in graph.Nodes)
        {
            if (n.Inputs.Length != 1 || n.Outputs.Length != 1 || n.Writes.Length > 0)
            {
                continue;
            }

            var inner = rewriter.ProducerOf(n.Inputs[0]);
            if (inner is null || inner.Op != n.Op || inner.Inputs.Length != 1)
            {
                continue;
            }

            var (_, innerAttributes) = rewriter.OpOf(inner);
            var source = rewriter.InputsOf(inner)[0];
            switch (n.Op.Name)
            {
                case "Reshape":
                    CollapseReshape(n, source, rewriter);
                    break;
                case "Permute":
                    CollapsePermute(n, innerAttributes, source, rewriter);
                    break;
                case "Slice":
                    CollapseSlice(n, innerAttributes, source, rewriter);
                    break;
            }
        }

        return rewriter.Build();
    }

    private static void CollapseReshape(Node n, Value source, GraphRewriter rewriter)
    {
        // The outer reshape's own attributes usually describe the target directly; otherwise spell the target out.
        if (rewriter.TryRewrite(n, n.Op, n.Attributes, [source]))
        {
            return;
        }

        var target = n.Output.Shape;
        if (target.Rank == 0)
        {
            return;
        }

        var inner = target.Dims.Skip(1).Select(d => d.IsSymbolic ? -1L : d.Fixed).ToArray();
        if (inner.Contains(-1L))
        {
            return;
        }

        if (!target[0].IsSymbolic)
        {
            rewriter.TryRewrite(n, n.Op, Attributes.From(("shape", (long[])[target[0].Fixed, .. inner])), [source]);
            return;
        }

        bool keepsOuter = source.Rank > 0 && source.Shape[0] == target[0];
        var attributes = keepsOuter
            ? Attributes.From(("shape", (long[])[0, .. inner]))
            : Attributes.From(("shape", (long[])[0, .. inner]), ("outer_symbol", target[0].Symbol!));
        rewriter.TryRewrite(n, n.Op, attributes, [source]);
    }

    private static void CollapsePermute(Node n, Attributes innerAttributes, Value source, GraphRewriter rewriter)
    {
        var outer = n.Attributes.Get<int[]>("perm");
        var first = innerAttributes.Get<int[]>("perm");
        if (outer.Length != first.Length)
        {
            return;
        }

        var composed = outer.Select(p => first[p]).ToArray();
        rewriter.TryRewrite(n, n.Op, n.Attributes.With("perm", composed), [source]);
    }

    private static void CollapseSlice(Node n, Attributes innerAttributes, Value source, GraphRewriter rewriter)
    {
        int rank = source.Rank;
        int axis = Normalize(n.Attributes.Get<int>("axis"), rank);
        int innerAxis = Normalize(innerAttributes.Get<int>("axis"), rank);
        if (axis != innerAxis || source.Shape[axis].IsSymbolic)
        {
            return;
        }

        long start = innerAttributes.Get<long>("start") + n.Attributes.Get<long>("start");
        long length = n.Attributes.Get<long>("length");
        long innerEnd = innerAttributes.Get<long>("start") + innerAttributes.Get<long>("length");
        if (start < 0 || start + length > innerEnd)
        {
            return;
        }

        var attributes = Attributes.From(("axis", axis), ("start", start), ("length", length));
        rewriter.TryRewrite(n, n.Op, attributes, [source]);
    }

    private static int Normalize(int axis, int rank) => axis < 0 ? axis + rank : axis;
}
