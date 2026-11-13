using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Expert selection of a mixture-of-experts block: router logits → scores → top-k experts per token and their mixing
/// weights. The single place where routing variants live (score function, selection bias, renormalization, scaling; later
/// group-limited top-k).
/// </summary>
internal static class MoeRouter
{
    // Keeps the renormalization finite when every selected score underflows to zero.
    private const float NormalizeEpsilon = 1e-20f;

    /// <summary>
    /// Routes the rows of <paramref name="h"/> [T, H] with the router weight <c>{prefix}router.weight</c> [E, H]. With the optional
    /// selection bias <c>{prefix}router_bias</c> [E] the bias only decides which experts are selected; the mixing weights are the
    /// unbiased scores of the selected experts.
    /// </summary>
    /// <returns>Mixing weights f32 [T, k] and expert indices i32 [T, k].</returns>
    /// <exception cref="InvalidDataException">The options do not describe a usable configuration.</exception>
    public static (Value Weights, Value Ids) Route(GraphBuilder b, Value h, string prefix, MoeOptions options)
    {
        if (!options.IsValid)
        {
            throw new InvalidDataException($"{prefix}: MoE weights present but moe.expert_count / moe.expert_used_count are missing or inconsistent.");
        }

        var logits = b.Linear(h, prefix + "router.weight");
        var scores = options.ScoreFunction switch
        {
            MoeScoreFunction.Sigmoid => b.Sigmoid(logits, $"{prefix}router_probs"),
            _ => b.Emit1("Softmax", [logits], null, $"{prefix}router_probs"),
        };
        var bias = b.TryWeight(prefix + "router_bias");
        var top = b.Emit("TopK", [bias is null ? scores : b.Add(scores, bias)], Attributes.From(("k", (long)options.ExpertsUsed)), $"{prefix}router_topk");
        var ids = top[1];
        var weights = bias is null ? top[0] : PickScores(b, scores, ids, prefix, options);
        if (options.NormalizeWeights)
        {
            weights = b.Binary("Div", weights, b.Scale(b.Emit1("ReduceSum", [weights]), 1f, NormalizeEpsilon));
        }

        if (options.WeightsScale != 1f)
        {
            weights = b.Scale(weights, options.WeightsScale);
        }

        return (weights, ids);
    }

    // The scores [T, E] of the selected experts ids [T, k]: one-hot pick (ids == iota) instead of a gather along the expert axis.
    private static Value PickScores(GraphBuilder b, Value scores, Value ids, string prefix, MoeOptions options)
    {
        int experts = options.Experts;
        int k = options.ExpertsUsed;
        var iotaBytes = new byte[experts * 4];
        for (int i = 0; i < experts; i++)
        {
            BitConverter.TryWriteBytes(iotaBytes.AsSpan(i * 4, 4), (float)i);
        }

        var iota = b.Constant($"{prefix}expert_iota", DataType.F32, new Shape(experts), iotaBytes);
        var onehot = b.Binary("Equal", b.Reshape(b.Emit1("Cast", [ids], Attributes.From(("type", "f32"))), 0, k, 1), iota);   // [T, k, E]
        return b.Reshape(b.Emit1("ReduceSum", [b.Mul(onehot, b.Reshape(scores, 0, 1, experts))]), 0, k);                       // [T, k]
    }
}
