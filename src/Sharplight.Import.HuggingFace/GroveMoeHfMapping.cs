using System.Runtime.CompilerServices;
using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// GroveMoE (remote code <c>GroveMoeForCausalLM</c>, whose config says <c>qwen3_moe</c>, see <see cref="HfConfig.ModelType"/>): Qwen3-MoE layers
/// whose experts are selected by the top-k of the router logits and weighted by their softmax probabilities renormalized over the selection,
/// plus adjugate experts <c>mlp.chunk_experts.{j}</c>, one per pair of consecutive routed experts, that every slot also runs and whose summed
/// output is scaled by 0.05. The pair size and the scale are constants of the remote code (<c>num_experts_per_group</c>, the 0.05 of its
/// forward), as is the unused <c>expert_bias</c> buffer, which is dropped.
/// </summary>
internal sealed partial class GroveMoeHfMapping : HfMoeFamilyMapping
{
    // GroveMoeSparseMoeBlock: num_experts_per_group = 2 and final = 0.05 * small + large.
    private const int ExpertsPerGroup = 2;
    private const float AdjugateScale = 0.05f;

    // Adjugate expert tensors of one import, keyed by that import's collector (the mapping itself is shared).
    private static readonly ConditionalWeakTable<HfWeightCollector, HfPartJoiner> s_adjugate = [];

    [GeneratedRegex(@"^mlp\.chunk_experts\.(\d+)\.(gate|up|down)_proj\.weight$")]
    private static partial Regex AdjugateExpertName();

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["grove_moe"];

    /// <inheritdoc/>
    protected override MoeRouting Routing(HfConfig c)
        => new(c.Int("num_experts"), c.Int("num_experts_per_tok"), c.Int("moe_intermediate_size"), "softmax", Normalize: true);

    /// <inheritdoc/>
    protected override void AddFamilyHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        md["decoder.moe.adjugate_group_size"] = ExpertsPerGroup;
        md["decoder.moe.adjugate_scale"] = AdjugateScale;
    }

    /// <inheritdoc/>
    protected override bool TryMapFamilyTensor(HfConfig c, string layerPrefix, string suffix, WeightInfo w, HfWeightCollector collector)
    {
        if (suffix == "mlp.expert_bias")
        {
            return true;
        }

        if (AdjugateExpertName().Match(suffix) is not { Success: true } match)
        {
            return false;
        }

        s_adjugate.GetValue(collector, _ => HfPartJoiner.Experts())
            .Add($"{layerPrefix}mlp.adjugate_experts.{match.Groups[2].Value}.weight", int.Parse(match.Groups[1].Value), w);
        return true;
    }

    /// <inheritdoc/>
    protected override void FinishWeights(HfConfig c, HfWeightCollector collector)
    {
        base.FinishWeights(c, collector);
        if (s_adjugate.TryGetValue(collector, out var adjugate))
        {
            adjugate.JoinInto(collector, c.Int("num_experts") / ExpertsPerGroup);
            s_adjugate.Remove(collector);
        }
    }
}
