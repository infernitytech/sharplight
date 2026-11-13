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

    // Routed experts (plus the optional shared expert) through the shared MoE block.
    private Value MoeMlp(GraphBuilder b, Value h, string p)
    {
        var options = new MoeOptions(_hp.Experts, _hp.ExpertsUsed, _hp.SigmoidRouter ? MoeScoreFunction.Sigmoid : MoeScoreFunction.Softmax,
            _hp.NormalizeExpertWeights, _hp.ExpertWeightsScale);
        return MoeBlock.Build(b, h, p + "mlp.", options, _hp.NEmbd, t => ActOf(b, t));
    }

    private Value Mlp(GraphBuilder b, Value h, string p)
    {
        Value Act(Value t) => ActOf(b, t);
        var up = b.Linear(h, p + "mlp.up.weight", p + "mlp.up.bias");
        var act = _hp.Gated ? b.Mul(Act(b.Linear(h, p + "mlp.gate.weight", p + "mlp.gate.bias")), up) : Act(up);
        return b.Linear(act, p + "mlp.down.weight", p + "mlp.down.bias", $"{p}ffn_out");
    }
}
