using Sharplight.Graph;

namespace Sharplight.Architectures;

/// <summary>
/// Expert selection of a mixture-of-experts block: router logits → scores → top-k experts per token and their mixing
/// weights. The single place where routing variants live (score function, renormalization, scaling; later selection bias and
/// group-limited top-k).
/// </summary>
internal static class MoeRouter
{
    /// <summary>
    /// Routes the rows of <paramref name="h"/> [T, H] with the router weight <c>{prefix}router.weight</c> [E, H].
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
        var top = b.Emit("TopK", [scores], Attributes.From(("k", (long)options.ExpertsUsed)), $"{prefix}router_topk");
        var weights = top[0];
        if (options.NormalizeWeights)
        {
            weights = b.Binary("Div", weights, b.Emit1("ReduceSum", [weights]));
        }

        if (options.WeightsScale != 1f)
        {
            weights = b.Scale(weights, options.WeightsScale);
        }

        return (weights, top[1]);
    }
}
