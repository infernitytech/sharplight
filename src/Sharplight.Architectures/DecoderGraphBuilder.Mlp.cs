using Sharplight.Graph;

namespace Sharplight.Architectures;

internal sealed partial class DecoderGraphBuilder
{
    private Value ActOf(GraphBuilder b, Value t, string layerPrefix = "") => _activation.Activate(b, t, layerPrefix);

    // Routed experts (plus the optional shared expert) through the shared MoE block.
    private Value MoeMlp(GraphBuilder b, Value h, string p)
    {
        var options = new MoeOptions(_hp.Experts, _hp.ExpertsUsed, _hp.SigmoidRouter ? MoeScoreFunction.Sigmoid : MoeScoreFunction.Softmax,
            _hp.NormalizeExpertWeights, _hp.ExpertWeightsScale);
        return MoeBlock.Build(b, h, p, "mlp.", options, _hp.NEmbd, _activation);
    }

    private Value Mlp(GraphBuilder b, Value h, string p)
    {
        var up = b.Linear(h, p + "mlp.up.weight", p + "mlp.up.bias");
        var act = _hp.Gated
            ? _activation.ActivateGated(b, b.Linear(h, p + "mlp.gate.weight", p + "mlp.gate.bias"), up, p)
            : _activation.Activate(b, up, p);
        return b.Linear(act, p + "mlp.down.weight", p + "mlp.down.bias", $"{p}ffn_out");
    }
}
