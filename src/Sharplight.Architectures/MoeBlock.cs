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
/// <item>Optional selection bias <c>router_bias</c> [E] (see <see cref="MoeRouter"/>).</item>
/// <item><c>router.weight</c> [E, H] with an optional logit bias <c>router.bias</c> [E] (gpt-oss).</item>
/// <item>Experts, rank 3: <c>experts.up.weight</c> [E, I, H], <c>experts.down.weight</c> [E, H, I] and either
/// <c>experts.gate.weight</c> [E, I, H] or the fused <c>experts.gate_up.weight</c> [E, 2I, H] (gate rows first). Without a gate the
/// experts are ungated MLPs (act(up)). Optional per-expert biases (gpt-oss) <c>experts.{gate,up,gate_up,down}.bias</c> [E, N] are
/// added to the selected experts' projections.</item>
/// <item>Optional latent space (Maion-Coder, Muse Spark): with <c>latent_down.weight</c> [L, H] the routed experts read the rows projected to the
/// latent size L (their weights are then [E, I, L] and [E, L, I]) and their weighted sum is projected back by <c>latent_up.weight</c> [H, L]; the
/// router and the shared expert read the full rows. An optional <c>latent_in_norm.weight</c> [L] RMS-normalizes the latent rows (Muse Spark).</item>
/// <item>Optional <c>experts.out_norm.weight</c> [H or L]: every selected expert's output is multiplied by it and then RMS-normalized
/// (epsilon <c>post_norm_eps</c>) before the routing weights apply (Muse Spark).</item>
/// <item>Optional shared expert: <c>shared.{gate,up,down}.weight</c> (a gated MLP; without <c>shared.gate</c> an ungated
/// act(up) MLP), optionally scaled per token by
/// sigmoid(h · <c>shared.input_gate.weight</c>) with the gate weight [1, H], and by <c>moe.shared_scale</c>.</item>
/// </list>
/// </remarks>
internal static class MoeBlock
{
    /// <summary>Returns <c>true</c> when the layer at <paramref name="prefix"/> has a routed MoE block.</summary>
    public static bool IsPresent(GraphBuilder b, string prefix) => b.HasWeight(prefix + "router.weight");

    /// <summary>Emits the block for h [T, H] and returns its output [T, H].</summary>
    /// <param name="b">The graph builder.</param>
    /// <param name="h">Normalized input rows [T, H].</param>
    /// <param name="layerPrefix">Weight-name prefix of the layer (<c>layers.{i}.</c>), which locates per-layer activation parameters.</param>
    /// <param name="block">Name of the block below the layer, ending in a dot (e.g. <c>mlp.</c>).</param>
    /// <param name="options">Routing configuration.</param>
    /// <param name="hidden">The hidden size H.</param>
    /// <param name="activation">The expert activation: <see cref="IMlpActivation.ActivateGated"/> for gated experts,
    /// <see cref="IMlpActivation.Activate"/> on up for ungated ones.</param>
    /// <param name="routerInput">Rows [T, H] the router reads when they differ from <paramref name="h"/> (SmallThinker routes on the
    /// layer input before the norm); null = <paramref name="h"/>.</param>
    /// <param name="sharedActivation">The shared expert's activation when it differs from the routed experts' (per-branch SwiGLU clamps);
    /// null = <paramref name="activation"/>.</param>
    public static Value Build(GraphBuilder b, Value h, string layerPrefix, string block, MoeOptions options, int hidden, IMlpActivation activation,
        Value? routerInput = null, IMlpActivation? sharedActivation = null)
    {
        string prefix = layerPrefix + block;
        int k = options.ExpertsUsed;
        var (weights, ids) = MoeRouter.Route(b, routerInput ?? h, prefix, options);
        var slotWeights = b.Reshape(weights, 0, k, 1);                                                                           // [T, k, 1]
        var latentDown = b.TryWeight(prefix + "latent_down.weight");
        int width = latentDown is null ? hidden : (int)latentDown.Shape[0].Fixed;
        var rows = latentDown is null ? h : b.Linear(h, prefix + "latent_down.weight", name: $"{prefix}latent_in");                // [T, H or L]
        if (b.TryWeight(prefix + "latent_in_norm.weight") is { } latentNorm)
        {
            rows = b.RmsNorm(rows, latentNorm, options.NormEps);
        }

        var expertInput = options.WeightsBeforeExperts ? b.Mul(b.Reshape(rows, 0, 1, width), slotWeights) : rows;                // [T, (k,) H or L]
        var act = ExpertActivations(b, expertInput, prefix, ids, Gated);
        var down = WithExpertBias(b, b.Emit1("MatMulId", [act, b.Weight(prefix + "experts.down.weight"), ids], null, $"{prefix}experts_down"),
            prefix + "experts.down.bias", ids);                                                                                  // [T, k, H]
        if (b.TryWeight(prefix + "experts.out_norm.weight") is { } outNorm)
        {
            down = b.RmsNorm(b.Mul(down, outNorm), null, options.PostNormEps);
        }

        var weighted = options.WeightsBeforeExperts ? down : b.Mul(down, slotWeights);
        var summed = b.Emit1("ReduceSum", [b.Permute(weighted, 0, 2, 1)]);                                                       // [T, H or L, 1]
        var routed = b.Reshape(summed, 0, width);
        if (latentDown is not null)
        {
            routed = b.Linear(routed, prefix + "latent_up.weight", name: $"{prefix}latent_out");                                   // [T, H]
        }

        if (!b.HasWeight(prefix + "shared.up.weight"))
        {
            return routed;
        }

        var shared = SharedExpert(b, h, prefix, (gate, up) => Activate(sharedActivation ?? activation, gate, up));
        if (options.SharedScale != 1f)
        {
            shared = b.Scale(shared, options.SharedScale);
        }

        return b.Add(routed, shared, $"{prefix}moe_out");

        // Activation of an MLP, gated when a gate projection exists.
        Value Gated(Value? gate, Value up) => Activate(activation, gate, up);

        Value Activate(IMlpActivation function, Value? gate, Value up)
            => gate is null ? function.Activate(b, up, layerPrefix) : function.ActivateGated(b, gate, up, layerPrefix);
    }

    // Gated (separate or fused gate/up) or ungated expert MLP inputs to the down projection: [T, k, I].
    private static Value ExpertActivations(GraphBuilder b, Value h, string prefix, Value ids, Func<Value?, Value, Value> activation)
    {
        if (b.TryWeight(prefix + "experts.gate_up.weight") is { } gateUpWeight)
        {
            long width = gateUpWeight.Shape[1].Fixed / 2;
            var gateUp = WithExpertBias(b, b.Emit1("MatMulId", [h, gateUpWeight, ids], null, $"{prefix}experts_gate_up"),
                prefix + "experts.gate_up.bias", ids);                                                                         // [T, k, 2I]
            return activation(b.Slice(gateUp, 2, 0, width), b.Slice(gateUp, 2, width, width));
        }

        var up = WithExpertBias(b, b.Emit1("MatMulId", [h, b.Weight(prefix + "experts.up.weight"), ids], null, $"{prefix}experts_up"),
            prefix + "experts.up.bias", ids);
        if (b.TryWeight(prefix + "experts.gate.weight") is not { } gateWeight)
        {
            return activation(null, up);
        }

        var gate = WithExpertBias(b, b.Emit1("MatMulId", [h, gateWeight, ids], null, $"{prefix}experts_gate"),
            prefix + "experts.gate.bias", ids);
        return activation(gate, up);
    }

    // Adds the selected experts' rows of an optional per-expert bias [E, N] to their projections [T, k, N].
    private static Value WithExpertBias(GraphBuilder b, Value projection, string biasName, Value ids)
    {
        if (b.TryWeight(biasName) is not { } bias)
        {
            return projection;
        }

        return b.Add(projection, b.Emit1("GatherAxis", [bias, ids], Attributes.From(("axis", 0L))));
    }

    private static Value SharedExpert(GraphBuilder b, Value h, string prefix, Func<Value?, Value, Value> activation)
    {
        var up = b.Linear(h, prefix + "shared.up.weight");
        var gate = b.HasWeight(prefix + "shared.gate.weight") ? b.Linear(h, prefix + "shared.gate.weight") : null;
        var act = activation(gate, up);
        var shared = b.Linear(act, prefix + "shared.down.weight", name: $"{prefix}shared_out");
        if (!b.HasWeight(prefix + "shared.input_gate.weight"))
        {
            return shared;
        }

        var inputGate = b.Sigmoid(b.Linear(h, prefix + "shared.input_gate.weight"), $"{prefix}shared_gate");                   // [T, 1]
        return b.Mul(shared, inputGate, $"{prefix}shared_gated");
    }
}
