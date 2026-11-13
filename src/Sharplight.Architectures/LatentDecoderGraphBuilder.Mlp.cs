using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

internal sealed partial class LatentDecoderGraphBuilder
{
    // Selection scores of experts outside the chosen groups: low enough never to win a top-k slot.
    private const float ExcludedGroupScore = -1e30f;

    /// <summary>Dense MLP from the weights below <paramref name="p"/> + <paramref name="block"/> (gated unless <c>mlp.gated</c> is false).</summary>
    private Value Mlp(GraphBuilder b, Value h, string p, string block, string name)
    {
        var up = b.Linear(h, p + block + "up.weight");
        var act = _hp.Gated ? _activation.ActivateGated(b, b.Linear(h, p + block + "gate.weight"), up, p) : _activation.Activate(b, up, p);
        return b.Linear(act, p + block + "down.weight", name: name);
    }

    /// <summary>
    /// Routed experts plus shared experts. Without expert groups this is the shared <see cref="MoeBlock"/>; group-limited routing
    /// (DeepSeek-V2 group_limited_greedy, V3 noaux_tc with several groups) is built here until <see cref="MoeOptions"/> has it.
    /// </summary>
    private Value Experts(GraphBuilder b, Value h, string p)
    {
        if (!_hp.GroupLimited)
        {
            return MoeBlock.Build(b, h, p, "mlp.", _hp.Moe, _hp.HiddenSize, _activation);
        }

        var moe = _hp.Moe;
        if (!moe.IsValid)
        {
            throw new InvalidDataException("latent_decoder: MoE weights present but moe.expert_count / moe.expert_used_count are missing or inconsistent.");
        }

        int k = moe.ExpertsUsed;
        var logits = b.Linear(h, p + "mlp.router.weight");
        var scores = moe.ScoreFunction == MoeScoreFunction.Sigmoid
            ? b.Sigmoid(logits, $"{p}router_scores")
            : b.Emit1("Softmax", [logits], null, $"{p}router_scores");                                                     // [T, E]
        var bias = b.TryWeight(p + "mlp.router_bias");
        var choice = LimitToBestGroups(b, bias is null ? scores : b.Add(scores, bias), p);
        var ids = b.Emit("TopK", [choice], Attributes.From(("k", (long)k)), $"{p}router_topk")[1];                          // [T, k] i32
        var weights = b.Emit1("GatherElements", [scores, ids], Attributes.From(("axis", 1L)));
        if (moe.NormalizeWeights)
        {
            weights = b.Binary("Div", weights, b.Scale(b.Emit1("ReduceSum", [weights]), 1f, 1e-20f));
        }

        if (moe.WeightsScale != 1f)
        {
            weights = b.Scale(weights, moe.WeightsScale);
        }

        var up = b.Emit1("MatMulId", [h, b.Weight(p + "mlp.experts.up.weight"), ids], null, $"{p}experts_up");
        var gate = b.Emit1("MatMulId", [h, b.Weight(p + "mlp.experts.gate.weight"), ids], null, $"{p}experts_gate");
        var down = b.Emit1("MatMulId", [_activation.ActivateGated(b, gate, up, p), b.Weight(p + "mlp.experts.down.weight"), ids], null, $"{p}experts_down");
        var weighted = b.Mul(down, b.Reshape(weights, 0, k, 1));                                                            // [T, k, H]
        var routed = b.Reshape(b.Emit1("ReduceSum", [b.Permute(weighted, 0, 2, 1)]), 0, _hp.HiddenSize);
        return b.HasWeight(p + "mlp.shared.up.weight") ? b.Add(routed, Mlp(b, h, p, "mlp.shared.", $"{p}shared_out")) : routed;
    }

    /// <summary>
    /// Group-limited routing: the experts form <c>moe.group_count</c> equal groups scored by their best (max) or two best (sum)
    /// selection scores; experts outside the <c>moe.group_used_count</c> best groups are excluded from the top-k.
    /// </summary>
    private Value LimitToBestGroups(GraphBuilder b, Value choice, string p)
    {
        int groups = _hp.ExpertGroups;
        int perGroup = _hp.Moe.Experts / groups;
        var grouped = b.Reshape(choice, 0, groups, perGroup);                                                                 // [T, G, E/G]
        var groupScores = _hp.GroupScore == "max"
            ? b.Emit1("ReduceMax", [grouped])
            : b.Emit1("ReduceSum", [b.Emit("TopK", [grouped], Attributes.From(("k", 2L)))[0]]);                               // [T, G, 1]
        var bestGroups = b.Emit("TopK", [b.Reshape(groupScores, 0, groups)], Attributes.From(("k", (long)_hp.ExpertGroupsUsed)), $"{p}router_groups")[1];
        var groupIota = new byte[groups * 4];
        for (int g = 0; g < groups; g++)
        {
            BitConverter.TryWriteBytes(groupIota.AsSpan(g * 4, 4), (float)g);
        }

        var chosen = b.Binary("Equal",
            b.Reshape(b.Emit1("Cast", [bestGroups], Attributes.From(("type", "f32"))), 0, _hp.ExpertGroupsUsed, 1),
            b.Constant($"{p}group_iota", DataType.F32, new Shape(groups), groupIota));                                      // [T, Gu, G]
        var mask = b.Emit1("ReduceSum", [b.Permute(chosen, 0, 2, 1)]);                                                      // [T, G, 1]: 1 = kept
        var penalty = b.Scale(mask, -ExcludedGroupScore, ExcludedGroupScore);                                               // 0 kept, -1e30 excluded
        return b.Reshape(b.Add(grouped, penalty), 0, _hp.Moe.Experts);
    }
}
