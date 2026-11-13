using System.Text;
using Sharplight.Tensors;

namespace Sharplight.Graph.Optimization;

/// <summary>
/// Merges nodes that apply the same op with equal attributes to the same operands (the later node's uses take the earlier
/// node's outputs), and constants with equal type, shape and bytes. Nodes that read or write state, own data-dependent symbols
/// or run subgraphs are never merged, and a node whose output is a graph output is never the one that is dropped.
/// </summary>
public sealed class CommonSubexpressionElimination : IGraphPass
{
    /// <inheritdoc/>
    public string Name => "common-subexpressions";

    /// <inheritdoc/>
    public bool RepeatUntilStable => true;

    /// <inheritdoc/>
    public ComputeGraph Run(ComputeGraph graph)
    {
        var rewriter = new GraphRewriter(graph);
        var operandIds = new Dictionary<Value, int>(ReferenceEqualityComparer.Instance);
        var constants = new Dictionary<(DataType Type, string Shape, int Hash), List<Value>>();
        var first = new Dictionary<string, Node>(StringComparer.Ordinal);
        var key = new StringBuilder();

        foreach (var n in graph.Nodes)
        {
            foreach (var input in n.Inputs)
            {
                if (input.Kind == ValueKind.Constant)
                {
                    MergeConstant(input, constants, rewriter);
                }
            }

            if (GraphFacts.IsPinned(n, graph))
            {
                continue;
            }

            key.Clear();
            key.Append(n.Op.Name).Append('|').Append(n.Attributes.ToString());
            // The textual form does not distinguish 1 from 1L or 1f: the CLR types are part of the key.
            foreach (var value in n.Attributes.Values)
            {
                key.Append(',').Append(value.GetType().Name);
            }

            foreach (var input in n.Inputs)
            {
                var operand = rewriter.Resolve(input);
                if (!operandIds.TryGetValue(operand, out int id))
                {
                    operandIds[operand] = id = operandIds.Count;
                }

                key.Append('|').Append(id);
            }

            string text = key.ToString();
            if (!first.TryGetValue(text, out var earlier))
            {
                first[text] = n;
                continue;
            }

            if (!n.Outputs.Select((o, k) => rewriter.CanSubstitute(o, earlier.Outputs[k])).All(ok => ok))
            {
                continue;
            }

            for (int k = 0; k < n.Outputs.Length; k++)
            {
                rewriter.Substitute(n.Outputs[k], earlier.Outputs[k]);
            }

            rewriter.Remove(n);
        }

        return rewriter.Build();
    }

    private static void MergeConstant(Value constant, Dictionary<(DataType, string, int), List<Value>> constants, GraphRewriter rewriter)
    {
        if (rewriter.Resolve(constant) != constant || constant.ConstantData is not { } data)
        {
            return;
        }

        var hash = new HashCode();
        hash.AddBytes(data);
        var bucketKey = (constant.Type, constant.Shape.ToString(), hash.ToHashCode());
        if (!constants.TryGetValue(bucketKey, out var bucket))
        {
            constants[bucketKey] = bucket = [];
        }

        foreach (var candidate in bucket)
        {
            if (ReferenceEquals(candidate, constant))
            {
                return;
            }

            if (candidate.ConstantData!.AsSpan().SequenceEqual(data) && rewriter.CanSubstitute(constant, candidate))
            {
                rewriter.Substitute(constant, candidate);
                return;
            }
        }

        bucket.Add(constant);
    }
}
