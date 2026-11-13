using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

internal sealed partial class LatentDecoderGraphBuilder
{
    // Selection scores of experts outside the chosen groups: low enough never to win a top-k slot.
    private const float ExcludedGroupScore = -1e30f;

    private Value Activation(GraphBuilder b, Value t) => _hp.Activation switch
    {
        "silu" => b.Silu(t),
        "gelu" => b.Unary("Gelu", t),
        "relu2" => Square(b, b.Unary("Relu", t)),
        _ => throw new NotSupportedException($"latent_decoder: unknown mlp.activation '{_hp.Activation}'."),
    };

    private static Value Square(GraphBuilder b, Value t) => b.Mul(t, t);

    /// <summary>Dense MLP from the weights below <paramref name="p"/> + <paramref name="block"/> (gated unless <c>mlp.gated</c> is false).</summary>
    private Value Mlp(GraphBuilder b, Value h, string p, string block, string name)
    {
        var up = b.Linear(h, p + block + "up.weight");
        var act = _hp.Gated ? b.Mul(Activation(b, b.Linear(h, p + block + "gate.weight")), up) : Activation(b, up);
        return b.Linear(act, p + block + "down.weight", name: name);
    }

    /// <summary>
    /// Routed experts: router scores (softmax or sigmoid), optional selection bias and group-limited choice, top-k experts per token,
    /// expert-indexed gated MLPs weighted by the (renormalized, scaled) unbiased scores, plus the shared experts.
    /// </summary>
    private Value Experts(GraphBuilder b, Value h, string p)
    {
        if (_hp.Experts <= 0 || _hp.ExpertsUsed <= 0)
        {
            throw new InvalidDataException("latent_decoder: MoE weights present but moe.expert_count / moe.expert_used_count are missing.");
        }

        int k = _hp.ExpertsUsed;
        var logits = b.Linear(h, p + "mlp.router.weight");
        var scores = _hp.SigmoidRouter ? b.Sigmoid(logits, $"{p}router_scores") : b.Emit1("Softmax", [logits], null, $"{p}router_scores");   // [T, E]
        var bias = b.TryWeight(p + "mlp.router.bias");
        var choice = bias is null ? scores : b.Add(scores, bias);
        if (_hp.GroupLimited)
        {
            choice = LimitToBestGroups(b, choice, p);
        }

        var top = b.Emit("TopK", [choice], Attributes.From(("k", (long)k)), $"{p}router_topk");
        var ids = top[1];                                                                                                   // [T, k] i32
        var weights = bias is null && !_hp.GroupLimited ? top[0] : b.Emit1("GatherElements", [scores, ids], Attributes.From(("axis", 1L)));
        if (_hp.NormalizeExpertWeights)
        {
            weights = b.Binary("Div", weights, b.Scale(b.Emit1("ReduceSum", [weights]), 1f, 1e-20f));
        }

        if (_hp.ExpertWeightsScale != 1f)
        {
            weights = b.Scale(weights, _hp.ExpertWeightsScale);
        }

        var up = b.Emit1("MatMulId", [h, b.Weight(p + "mlp.experts.up.weight"), ids], null, $"{p}experts_up");
        var gate = b.Emit1("MatMulId", [h, b.Weight(p + "mlp.experts.gate.weight"), ids], null, $"{p}experts_gate");
        var down = b.Emit1("MatMulId", [b.Mul(Activation(b, gate), up), b.Weight(p + "mlp.experts.down.weight"), ids], null, $"{p}experts_down"); // [T, k, H]
        var weighted = b.Mul(down, b.Reshape(weights, 0, k, 1));
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
        int perGroup = _hp.Experts / groups;
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
        return b.Reshape(b.Add(grouped, penalty), 0, _hp.Experts);
    }
}
