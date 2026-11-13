using Sharplight.Graph;

namespace Sharplight.Architectures;

internal sealed partial class DecoderGraphBuilder
{
    private Value ActOf(GraphBuilder b, Value t) => _hp.Activation switch
    {
        "silu" => b.Silu(t),
        "gelu" => b.Unary("Gelu", t),
        "gelu_tanh" => b.Unary("GeluTanh", t),
        "relu" => b.Unary("Relu", t),
        _ => throw new NotSupportedException($"decoder: unknown mlp.activation '{_hp.Activation}'."),
    };

    // Softmax router, top-k experts per token, expert-indexed gate/up/down products, weighted sum over the k slots.
    private Value MoeMlp(GraphBuilder b, Value h, string p)
    {
        if (_hp.Experts <= 0 || _hp.ExpertsUsed <= 0)
        {
            throw new InvalidDataException("decoder: MoE weights present but moe.expert_count / moe.expert_used_count are missing.");
        }

        var logits = b.Linear(h, p + "mlp.router.weight");
        var probs = _hp.SigmoidRouter ? b.Sigmoid(logits, $"{p}router_probs") : b.Emit1("Softmax", [logits], null, $"{p}router_probs");
        var top = b.Emit("TopK", [probs], Attributes.From(("k", (long)_hp.ExpertsUsed)), $"{p}router_topk");
        Value weights = top[0];
        Value ids = top[1];
        if (_hp.NormalizeExpertWeights)
        {
            weights = b.Binary("Div", weights, b.Emit1("ReduceSum", [weights]));
        }

        var up = b.Emit1("MatMulId", [h, b.Weight(p + "mlp.experts.up.weight"), ids], null, $"{p}experts_up");
        var gate = b.Emit1("MatMulId", [h, b.Weight(p + "mlp.experts.gate.weight"), ids], null, $"{p}experts_gate");
        var act = b.Mul(ActOf(b, gate), up);
        var down = b.Emit1("MatMulId", [act, b.Weight(p + "mlp.experts.down.weight"), ids], null, $"{p}experts_down");   // [T, k, H]
        if (_hp.ExpertWeightsScale != 1f)
        {
            weights = b.Scale(weights, _hp.ExpertWeightsScale);
        }

        var weighted = b.Mul(down, b.Reshape(weights, 0, _hp.ExpertsUsed, 1));
        var summed = b.Emit1("ReduceSum", [b.Permute(weighted, 0, 2, 1)]);                                              // [T, H, 1]
        var routed = b.Reshape(summed, 0, _hp.NEmbd);
        if (b.HasWeight(p + "mlp.shared.up.weight"))
        {
            var sUp = b.Linear(h, p + "mlp.shared.up.weight");
            var sAct = b.Mul(ActOf(b, b.Linear(h, p + "mlp.shared.gate.weight")), sUp);
            routed = b.Add(routed, b.Linear(sAct, p + "mlp.shared.down.weight", name: $"{p}shared_out"));
        }
        return routed;
    }

    private Value Mlp(GraphBuilder b, Value h, string p)
    {
        Value Act(Value t) => ActOf(b, t);
        var up = b.Linear(h, p + "mlp.up.weight", p + "mlp.up.bias");
        var act = _hp.Gated ? b.Mul(Act(b.Linear(h, p + "mlp.gate.weight", p + "mlp.gate.bias")), up) : Act(up);
        return b.Linear(act, p + "mlp.down.weight", p + "mlp.down.bias", $"{p}ffn_out");
    }
}
