using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Expert selection of a mixture-of-experts block: router logits → scores → top-k experts per token and their mixing
/// weights. The single place where routing variants live (score function, logit bias, selection bias, group-limited top-k,
/// renormalization, scaling); see <see cref="MoeOptions"/> for the order in which they apply.
/// </summary>
internal static class MoeRouter
{
    // Keeps the renormalization finite when every selected score underflows to zero.
    private const float NormalizeEpsilon = 1e-20f;

    // Selection score of the experts in groups that were not chosen: below any real score, but finite for every top-k kernel.
    private const float MaskedScore = float.MinValue;

    /// <summary>
    /// Routes the rows of <paramref name="h"/> [T, H] with the router weight <c>{prefix}router.weight</c> [E, H] and the optional
    /// logit bias <c>{prefix}router.bias</c> [E]. With the optional selection bias <c>{prefix}router_bias</c> [E] the bias only decides
    /// which experts are selected; the mixing weights are the unbiased scores of the selected experts.
    /// </summary>
    /// <returns>Mixing weights f32 [T, k] and expert indices i32 [T, k].</returns>
    /// <exception cref="InvalidDataException">The options do not describe a usable configuration.</exception>
    public static (Value Weights, Value Ids) Route(GraphBuilder b, Value h, string prefix, MoeOptions options)
    {
        if (!options.IsValid)
        {
            throw new InvalidDataException($"{prefix}: MoE weights present but moe.expert_count / moe.expert_used_count / moe.group_count are missing or inconsistent.");
        }

        var logits = b.Linear(h, prefix + "router.weight", prefix + "router.bias");
        var scores = Scores(b, logits, prefix, options.ScoreFunction);
        var bias = b.TryWeight(prefix + "router_bias");
        var selection = bias is null ? scores : b.Add(scores, bias);
        if (options.IsGroupLimited)
        {
            selection = LimitToBestGroups(b, selection, prefix, options);
        }

        var top = b.Emit("TopK", [selection], Attributes.From(("k", (long)options.ExpertsUsed)), $"{prefix}router_topk");
        var ids = top[1];
        var weights = bias is null && !options.IsGroupLimited ? top[0] : PickScores(b, scores, ids, prefix, options);
        if (options.ScoreFunction == MoeScoreFunction.SoftmaxSelected)
        {
            weights = b.Emit1("Softmax", [weights], null, $"{prefix}router_weights");
        }

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

    // Expert scores [T, E] from the router logits. Softmax-after-top-k keeps the logits: the softmax runs on the selected ones.
    private static Value Scores(GraphBuilder b, Value logits, string prefix, MoeScoreFunction function) => function switch
    {
        MoeScoreFunction.Sigmoid => b.Sigmoid(logits, $"{prefix}router_probs"),
        MoeScoreFunction.SoftmaxSelected => logits,
        MoeScoreFunction.SqrtSoftplus => b.Unary("Sqrt", b.Softplus(logits), $"{prefix}router_probs"),
        _ => b.Emit1("Softmax", [logits], null, $"{prefix}router_probs"),
    };

    /// <summary>
    /// Group-limited routing (DeepSeek-V3 <c>n_group</c> / <c>topk_group</c>): the experts form <c>Groups</c> contiguous groups, each group
    /// is ranked by its best selection scores, and the experts outside the best <c>GroupsUsed</c> groups are masked out of the top-k.
    /// </summary>
    private static Value LimitToBestGroups(GraphBuilder b, Value selection, string prefix, MoeOptions options)
    {
        int groups = options.Groups;
        int perGroup = options.Experts / groups;
        var grouped = b.Reshape(selection, 0, groups, perGroup);                                                                // [T, G, n]
        Value groupScores = options.GroupScore == MoeGroupScore.Max
            ? b.Emit1("ReduceMax", [grouped])
            : b.Emit1("ReduceSum", [b.Emit("TopK", [grouped], Attributes.From(("k", 2L)), $"{prefix}router_group_top2")[0]]);  // [T, G, 1]
        var bestGroups = b.Emit("TopK", [b.Reshape(groupScores, 0, groups)], Attributes.From(("k", (long)options.GroupsUsed)), $"{prefix}router_group_topk")[1];
        var chosen = b.Reshape(OneHotCount(b, bestGroups, groups, options.GroupsUsed, $"{prefix}group_iota"), 0, groups, 1);    // [T, G, 1]
        var masked = b.Emit1("Where", [chosen, grouped, ScalarConstant(b, $"{prefix}group_masked", MaskedScore)]);
        return b.Reshape(masked, 0, options.Experts);
    }

    // The scores [T, E] of the selected experts ids [T, k]: one-hot pick (ids == iota) instead of a gather along the expert axis.
    private static Value PickScores(GraphBuilder b, Value scores, Value ids, string prefix, MoeOptions options)
    {
        int experts = options.Experts;
        int k = options.ExpertsUsed;
        var onehot = OneHot(b, ids, experts, k, $"{prefix}expert_iota");                                                        // [T, k, E]
        return b.Reshape(b.Emit1("ReduceSum", [b.Mul(onehot, b.Reshape(scores, 0, 1, experts))]), 0, k);                        // [T, k]
    }

    // f32 one-hot rows [T, k, n] of the indices [T, k] (1 where ids[t, s] == j).
    private static Value OneHot(GraphBuilder b, Value ids, int n, int k, string iotaName)
    {
        var iotaBytes = new byte[n * 4];
        for (int i = 0; i < n; i++)
        {
            BitConverter.TryWriteBytes(iotaBytes.AsSpan(i * 4, 4), (float)i);
        }

        var iota = b.Constant(iotaName, DataType.F32, new Shape(n), iotaBytes);
        return b.Binary("Equal", b.Reshape(b.Emit1("Cast", [ids], Attributes.From(("type", "f32"))), 0, k, 1), iota);
    }

    // How often each of n classes occurs among the indices [T, k]: [T, n, 1] (0 or 1 for distinct indices).
    private static Value OneHotCount(GraphBuilder b, Value ids, int n, int k, string iotaName)
        => b.Emit1("ReduceSum", [b.Permute(OneHot(b, ids, n, k, iotaName), 0, 2, 1)]);

    private static Value ScalarConstant(GraphBuilder b, string name, float value)
        => b.Constant(name, DataType.F32, new Shape(1), BitConverter.GetBytes(value));
}
