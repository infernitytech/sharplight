using Sharplight.Graph;

namespace Sharplight.Architectures;

internal sealed partial class DecoderGraphBuilder
{
    private Value ActOf(GraphBuilder b, Value t, string layerPrefix = "") => _activation.Activate(b, t, layerPrefix);

    // Routed experts (plus the optional shared expert) through the shared MoE block.
    // routerInput: the layer input, for models whose router reads it instead of the normed rows (moe.router_input).
    private Value MoeMlp(GraphBuilder b, Value h, string p, Value layerInput)
        => MoeBlock.Build(b, h, p, "mlp.", _hp.Moe, _hp.NEmbd, _activation, _hp.RouterReadsLayerInput ? layerInput : null);

    private Value Mlp(GraphBuilder b, Value h, string p)
    {
        var up = b.Linear(h, p + "mlp.up.weight", p + "mlp.up.bias");
        var act = _hp.Gated
            ? _activation.ActivateGated(b, b.Linear(h, p + "mlp.gate.weight", p + "mlp.gate.bias"), up, p)
            : _activation.Activate(b, up, p);
        return b.Linear(SubNorm(b, act, p + "mlp.sub_norm.weight"), p + "mlp.down.weight", p + "mlp.down.bias", $"{p}ffn_out");
    }

    // BitNet sub-layer norm: an RMS norm of the input of the output projection (attention or MLP) when the layer has its weight.
    private Value SubNorm(GraphBuilder b, Value x, string weightName)
        => b.TryWeight(weightName) is { } weight ? b.RmsNorm(x, weight, _hp.Eps) : x;
}
