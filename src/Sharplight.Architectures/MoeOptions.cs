using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// Routing configuration of a mixture-of-experts feed-forward block (see <see cref="MoeBlock"/>), read from the canonical
/// <c>{arch}.moe.*</c> hyper-parameters that every MoE-capable architecture shares. New routing variants (selection bias,
/// group-limited top-k, ...) are added here and in <see cref="MoeRouter"/>, so all adapters pick them up.
/// </summary>
/// <param name="Experts">Number of routed experts (<c>moe.expert_count</c>).</param>
/// <param name="ExpertsUsed">Experts selected per token (<c>moe.expert_used_count</c>).</param>
/// <param name="ScoreFunction">Router scores from the router logits (<c>moe.router_activation</c>: "softmax" or "sigmoid").</param>
/// <param name="NormalizeWeights">Renormalize the selected scores to sum 1 (<c>moe.normalize_weights</c>).</param>
/// <param name="WeightsScale">Factor applied to the expert weights after normalization (<c>moe.weights_scale</c>).</param>
internal sealed record MoeOptions(int Experts, int ExpertsUsed, MoeScoreFunction ScoreFunction, bool NormalizeWeights, float WeightsScale)
{
    /// <summary>Reads the <c>moe.*</c> keys of <paramref name="source"/>.</summary>
    /// <param name="source">The model source.</param>
    /// <param name="normalizeDefault">The family's default for <c>moe.normalize_weights</c>.</param>
    /// <exception cref="InvalidDataException">The router activation is unknown.</exception>
    public static MoeOptions Read(ModelSource source, bool normalizeDefault = false)
    {
        string activation = source.Arch("moe.router_activation", "softmax");
        var score = activation switch
        {
            "softmax" => MoeScoreFunction.Softmax,
            "sigmoid" => MoeScoreFunction.Sigmoid,
            _ => throw new InvalidDataException($"{source.Architecture}: unknown moe.router_activation '{activation}'."),
        };
        return new MoeOptions(
            Experts: source.Arch("moe.expert_count", 0),
            ExpertsUsed: source.Arch("moe.expert_used_count", 0),
            ScoreFunction: score,
            NormalizeWeights: source.Arch("moe.normalize_weights", normalizeDefault),
            WeightsScale: source.Arch("moe.weights_scale", 1f));
    }

    /// <summary>Whether the counts describe a usable configuration.</summary>
    public bool IsValid => Experts > 0 && ExpertsUsed > 0 && ExpertsUsed <= Experts;
}
