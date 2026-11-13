namespace Sharplight.Graph.Optimization;

/// <summary>
/// Evaluates nodes whose operands are all graph constants (and operand-free <c>Fill</c> nodes) on the host and replaces their
/// output by a constant. Only ops whose reference (CPU) semantics are reproduced bit-exactly are folded (data movement, casts,
/// <c>Where</c>, <c>Fill</c> and exact f32 elementwise arithmetic, see <see cref="MaxOutputBytes"/> for the size cap); other ops,
/// symbolic shapes, graph outputs and nodes with side effects are left alone.
/// </summary>
public sealed class ConstantFolding : IGraphPass
{
    /// <summary>Default for <see cref="MaxOutputBytes"/>: 256 KiB.</summary>
    public const long DefaultMaxOutputBytes = 256 * 1024;

    /// <summary>
    /// Largest folded output (and so the most host memory and evaluation work spent per node); larger results stay computed on
    /// the device, so huge tensors are never materialized on the host or embedded in the graph.
    /// </summary>
    public long MaxOutputBytes { get; init; } = DefaultMaxOutputBytes;

    /// <inheritdoc/>
    public string Name => "constant-folding";

    /// <inheritdoc/>
    public bool RepeatUntilStable => true;

    /// <inheritdoc/>
    public ComputeGraph Run(ComputeGraph graph)
    {
        var rewriter = new GraphRewriter(graph);
        var folded = new Dictionary<Value, HostTensor>();
        foreach (var n in graph.Nodes)
        {
            if (n.Outputs.Length != 1 || GraphFacts.IsPinned(n, graph) || !ConstantEvaluator.Supports(n.Op.Name)
                || !GraphFacts.IsFullyStatic(n.Output.Shape) || n.Output.Type.ByteSize(n.Output.Shape.Elements()) > MaxOutputBytes)
            {
                continue;
            }

            var operands = new List<HostTensor>(n.Inputs.Length);
            foreach (var input in n.Inputs)
            {
                var v = rewriter.Resolve(input);
                if (!folded.TryGetValue(v, out var t))
                {
                    t = HostTensor.Of(v);
                    if (t is null)
                    {
                        break;
                    }

                    folded[v] = t;
                }

                operands.Add(t);
            }

            if (operands.Count != n.Inputs.Length)
            {
                continue;
            }

            var dims = n.Output.Shape.Dims.Select(d => d.Fixed).ToArray();
            if (ConstantEvaluator.Evaluate(n, operands, dims) is not { } data || data.Length != n.Output.Type.ByteSize(n.Output.Shape.Elements()))
            {
                continue;
            }

            var constant = rewriter.NewConstant(n.Output.Name, n.Output.Type, n.Output.Shape, data);
            if (!rewriter.CanSubstitute(n.Output, constant))
            {
                continue;
            }

            rewriter.Substitute(n.Output, constant);
            folded[constant] = new HostTensor(constant.Type, dims, data);
        }

        return rewriter.Build();
    }
}
