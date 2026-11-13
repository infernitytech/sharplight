using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// Routing configuration of a mixture-of-experts feed-forward block (see <see cref="MoeBlock"/>), read from the canonical
/// <c>{arch}.moe.*</c> hyper-parameters that every MoE-capable architecture shares. New routing variants are added here and in
/// <see cref="MoeRouter"/>, so all adapters pick them up.
/// </summary>
/// <remarks>
/// Routing order (see <see cref="MoeRouter"/>): logits = router · h (+ <c>router.bias</c>) → scores (<see cref="ScoreFunction"/>) →
/// selection scores = scores (+ selection bias <c>router_bias</c>) → optional group limit (<see cref="Groups"/> /
/// <see cref="GroupsUsed"/>) → top-k on the selection scores → weights = the unbiased scores of the selected experts → optional
/// renormalization (<see cref="NormalizeWeights"/>) → scale (<see cref="WeightsScale"/>, DeepSeek's <c>routed_scaling_factor</c>).
/// </remarks>
/// <param name="Experts">Number of routed experts (<c>moe.expert_count</c>).</param>
/// <param name="ExpertsUsed">Experts selected per token (<c>moe.expert_used_count</c>).</param>
/// <param name="ScoreFunction">Router scores from the router logits (<c>moe.router_activation</c>: "softmax" (default), "sigmoid",
/// "softmax_topk" (softmax over the selected logits, after the top-k) or "sqrt_softplus").</param>
/// <param name="NormalizeWeights">Renormalize the selected weights to sum 1 (<c>moe.normalize_weights</c>, HF <c>norm_topk_prob</c>).</param>
/// <param name="WeightsScale">Factor applied to the expert weights after normalization (<c>moe.weights_scale</c>, HF <c>routed_scaling_factor</c>).</param>
/// <param name="Groups">Expert groups for group-limited routing (<c>moe.group_count</c>, HF <c>n_group</c>); 1 = no group limit.</param>
/// <param name="GroupsUsed">Groups whose experts stay selectable per token (<c>moe.group_used_count</c>, HF <c>topk_group</c>).</param>
/// <param name="GroupScore">How a group is ranked (<c>moe.group_score</c>: "top2_sum" (default) or "max").</param>
/// <param name="WeightsBeforeExperts">Multiply each selected expert's input by its weight instead of its output
/// (<c>moe.weights_before_experts</c>, Llama 4).</param>
internal sealed record MoeOptions(
    int Experts, int ExpertsUsed, MoeScoreFunction ScoreFunction, bool NormalizeWeights, float WeightsScale,
    int Groups = 1, int GroupsUsed = 1, MoeGroupScore GroupScore = MoeGroupScore.TopTwoSum, bool WeightsBeforeExperts = false)
{
    /// <summary>Reads the <c>moe.*</c> keys of <paramref name="source"/>.</summary>
    /// <param name="source">The model source.</param>
    /// <param name="normalizeDefault">The family's default for <c>moe.normalize_weights</c>.</param>
    /// <exception cref="InvalidDataException">The router activation or group score is unknown.</exception>
    public static MoeOptions Read(ModelSource source, bool normalizeDefault = false)
    {
        string activation = source.Arch("moe.router_activation", "softmax");
        var score = activation switch
        {
            "softmax" => MoeScoreFunction.Softmax,
            "sigmoid" => MoeScoreFunction.Sigmoid,
            "softmax_topk" => MoeScoreFunction.SoftmaxSelected,
            "sqrt_softplus" => MoeScoreFunction.SqrtSoftplus,
            _ => throw new InvalidDataException($"{source.Architecture}: unknown moe.router_activation '{activation}'."),
        };
        string groupScore = source.Arch("moe.group_score", "top2_sum");
        var group = groupScore switch
        {
            "top2_sum" => MoeGroupScore.TopTwoSum,
            "max" => MoeGroupScore.Max,
            _ => throw new InvalidDataException($"{source.Architecture}: unknown moe.group_score '{groupScore}'."),
        };
        int groups = Math.Max(1, source.Arch("moe.group_count", 1));
        return new MoeOptions(
            Experts: source.Arch("moe.expert_count", 0),
            ExpertsUsed: source.Arch("moe.expert_used_count", 0),
            ScoreFunction: score,
            NormalizeWeights: source.Arch("moe.normalize_weights", normalizeDefault),
            WeightsScale: source.Arch("moe.weights_scale", 1f),
            Groups: groups,
            GroupsUsed: source.Arch("moe.group_used_count", groups),
            GroupScore: group,
            WeightsBeforeExperts: source.Arch("moe.weights_before_experts", false));
    }

    /// <summary>Whether the counts describe a usable configuration.</summary>
    public bool IsValid
        => Experts > 0 && ExpertsUsed > 0 && ExpertsUsed <= Experts
           && Groups >= 1 && Experts % Groups == 0 && GroupsUsed >= 1 && GroupsUsed <= Groups
           && ExpertsUsed <= GroupsUsed * (Experts / Groups);

    /// <summary>Whether routing is limited to the best <see cref="GroupsUsed"/> of several groups.</summary>
    public bool IsGroupLimited => Groups > 1 && GroupsUsed < Groups;
}
