using Sharplight.Graph;

namespace Sharplight.Architectures;

/// <summary>
/// The mixture-of-experts feed-forward block shared by the decoder adapters: routing (<see cref="MoeRouter"/>), expert-indexed
/// products (MatMulId) for the selected experts, the weighted sum over the k slots, and an optional shared expert that every token
/// passes through, optionally scaled by a sigmoid gate.
/// </summary>
/// <remarks>
/// Canonical weights below <c>prefix</c> (e.g. <c>layers.3.mlp.</c>):
/// <list type="bullet">
/// <item><c>router.weight</c> [E, H].</item>
/// <item>Experts, rank 3: <c>experts.up.weight</c> [E, I, H], <c>experts.down.weight</c> [E, H, I] and either
/// <c>experts.gate.weight</c> [E, I, H] or the fused <c>experts.gate_up.weight</c> [E, 2I, H] (gate rows first). Without a gate the
/// experts are ungated MLPs (act(up)).</item>
/// <item>Optional shared expert: <c>shared.{gate,up,down}.weight</c> (a gated MLP), optionally scaled per token by
/// sigmoid(h · <c>shared.input_gate.weight</c>) with the gate weight [1, H].</item>
/// </list>
/// </remarks>
internal static class MoeBlock
{
    /// <summary>Returns <c>true</c> when the layer at <paramref name="prefix"/> has a routed MoE block.</summary>
    public static bool IsPresent(GraphBuilder b, string prefix) => b.HasWeight(prefix + "router.weight");

    /// <summary>Emits the block for h [T, H] and returns its output [T, H].</summary>
    /// <param name="b">The graph builder.</param>
    /// <param name="h">Normalized input rows [T, H].</param>
    /// <param name="prefix">Weight-name prefix of the block, ending in a dot.</param>
    /// <param name="options">Routing configuration.</param>
    /// <param name="hidden">The hidden size H.</param>
    /// <param name="activation">The MLP activation (applied to the gate, or to up for ungated experts).</param>
    public static Value Build(GraphBuilder b, Value h, string prefix, MoeOptions options, int hidden, Func<Value, Value> activation)
    {
        var (weights, ids) = MoeRouter.Route(b, h, prefix, options);
        var act = ExpertActivations(b, h, prefix, ids, activation);
        var down = b.Emit1("MatMulId", [act, b.Weight(prefix + "experts.down.weight"), ids], null, $"{prefix}experts_down");   // [T, k, H]
        var weighted = b.Mul(down, b.Reshape(weights, 0, options.ExpertsUsed, 1));
        var summed = b.Emit1("ReduceSum", [b.Permute(weighted, 0, 2, 1)]);                                                       // [T, H, 1]
        var routed = b.Reshape(summed, 0, hidden);
        if (!b.HasWeight(prefix + "shared.up.weight"))
        {
            return routed;
        }

        return b.Add(routed, SharedExpert(b, h, prefix, activation), $"{prefix}moe_out");
    }

    // Gated (separate or fused gate/up) or ungated expert MLP inputs to the down projection: [T, k, I].
    private static Value ExpertActivations(GraphBuilder b, Value h, string prefix, Value ids, Func<Value, Value> activation)
    {
        if (b.TryWeight(prefix + "experts.gate_up.weight") is { } gateUpWeight)
        {
            long width = gateUpWeight.Shape[1].Fixed / 2;
            var gateUp = b.Emit1("MatMulId", [h, gateUpWeight, ids], null, $"{prefix}experts_gate_up");                       // [T, k, 2I]
            return b.Mul(activation(b.Slice(gateUp, 2, 0, width)), b.Slice(gateUp, 2, width, width));
        }

        var up = b.Emit1("MatMulId", [h, b.Weight(prefix + "experts.up.weight"), ids], null, $"{prefix}experts_up");
        if (b.TryWeight(prefix + "experts.gate.weight") is not { } gateWeight)
        {
            return activation(up);
        }

        var gate = b.Emit1("MatMulId", [h, gateWeight, ids], null, $"{prefix}experts_gate");
        return b.Mul(activation(gate), up);
    }

    private static Value SharedExpert(GraphBuilder b, Value h, string prefix, Func<Value, Value> activation)
    {
        var up = b.Linear(h, prefix + "shared.up.weight");
        var act = b.Mul(activation(b.Linear(h, prefix + "shared.gate.weight")), up);
        var shared = b.Linear(act, prefix + "shared.down.weight", name: $"{prefix}shared_out");
        if (!b.HasWeight(prefix + "shared.input_gate.weight"))
        {
            return shared;
        }

        var gate = b.Sigmoid(b.Linear(h, prefix + "shared.input_gate.weight"), $"{prefix}shared_gate");                        // [T, 1]
        return b.Mul(shared, gate, $"{prefix}shared_gated");
    }
}
