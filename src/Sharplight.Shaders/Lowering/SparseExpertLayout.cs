using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Runtime;

namespace Sharplight.Shaders.Lowering;

/// <summary>
/// What a segment compiled with <see cref="CompileOptions.SparseExpertSlots"/> holds sparsely: the expert ids input (rewritten to slots by
/// <see cref="SparseExpertSelection"/>), the experts of its banks, and the per-expert tables it gathers rows of by those ids (e.g. the
/// experts' biases), which the devices keep whole.
/// </summary>
/// <param name="Ids">The i32 [rows, slots] input of expert ids.</param>
/// <param name="Experts">The experts of every bank.</param>
/// <param name="Tables">The gathered per-expert tables.</param>
internal sealed record SparseExpertLayout(Value Ids, int Experts, IReadOnlySet<WeightInfo> Tables)
{
    /// <summary>
    /// The layout of <paramref name="segment"/>: every weight it reads must be the bank of a <c>MatMulId</c> (input 1, [experts, ...]) or a
    /// table of a <c>GatherAxis</c> on axis 0 ([experts, N]), indexed by one ids input of the segment.
    /// </summary>
    /// <exception cref="NotSupportedException">The segment reads other weights or more than one ids value.</exception>
    public static SparseExpertLayout Of(GraphSegment segment)
    {
        Value? ids = null;
        int experts = 0;
        HashSet<WeightInfo> tables = [];
        foreach (var n in segment.Nodes)
        {
            for (int i = 0; i < n.Inputs.Length; i++)
            {
                if (n.Inputs[i].Weight is not { } weight)
                {
                    continue;
                }

                bool bank = n.Op.Name == "MatMulId" && i == 1;
                bool table = n.Op.Name == "GatherAxis" && i == 0 && n.Attributes.Get("axis", 0L) == 0 && n.Inputs[0].Shape.Rank == 2;
                var nodeIds = n.Inputs[bank ? 2 : 1];
                if (!bank && !table || ids is not null && nodeIds != ids || experts > 0 && n.Inputs[i].Shape[0].Fixed != experts)
                {
                    throw new NotSupportedException($"Sparse expert streaming: {n.Op.Name} reads a weight that is not an expert bank of the segment's ids (output {n.Outputs[0].Name}).");
                }

                if (table)
                {
                    tables.Add(weight);
                }

                ids = nodeIds;
                experts = (int)n.Inputs[i].Shape[0].Fixed;
            }
        }

        if (ids is null || !segment.Inputs.Contains(ids) || ids.Type != Tensors.DataType.I32)
        {
            throw new NotSupportedException("Sparse expert streaming: the segment has no expert banks indexed by an i32 input.");
        }

        return new SparseExpertLayout(ids, experts, tables);
    }
}
