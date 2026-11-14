using Sharplight.Graph;
using Sharplight.Models;

namespace Sharplight.Architectures;

internal sealed partial class DecoderGraphBuilder
{
    private Value ActOf(GraphBuilder b, Value t, string layerPrefix = "") => _activation.Activate(b, t, layerPrefix);

    // Routed experts (plus the optional shared expert) through the shared MoE block.
    // routerInput: the layer input, for models whose router reads it instead of the normed rows (moe.router_input).
    private Value MoeMlp(GraphBuilder b, Value h, string p, int il, Value layerInput)
        => MoeBlock.Build(b, h, p, "mlp.", _hp.Moe, _hp.NEmbd, ExpertActivationOf(il), _hp.RouterReadsLayerInput ? layerInput : null,
            DenseActivationOf(il));

    private Value Mlp(GraphBuilder b, Value h, string p, int il)
    {
        var up = b.Linear(h, p + "mlp.up.weight", p + "mlp.up.bias");
        var activation = DenseActivationOf(il);
        var act = _hp.Gated
            ? activation.ActivateGated(b, SparseGate(b, b.Linear(h, p + "mlp.gate.weight", p + "mlp.gate.bias"), il, p), up, p)
            : activation.Activate(b, up, p);
        return b.Linear(SubNorm(b, act, p + "mlp.sub_norm.weight"), p + "mlp.down.weight", p + "mlp.down.bias", $"{p}ffn_out");
    }

    // Activation sparsity (Gemma 3n): with a target sparsity s for the layer, the gate keeps only what exceeds its row's
    // mean + std · Φ⁻¹(s) (population std), shifted down by that cutoff: relu(g − mean − std · Φ⁻¹(s)).
    private Value SparseGate(GraphBuilder b, Value gate, int il, string p)
    {
        if (il >= _hp.NLayer || _hp.ActivationSparsityOf(il) is not (> 0f and var sparsity))
        {
            return gate;
        }

        float inverse = 1f / gate.Shape[-1].Fixed;
        var centered = b.Sub(gate, b.Scale(b.Emit1("ReduceSum", [gate]), inverse));
        var std = b.Unary("Sqrt", b.Scale(b.Emit1("ReduceSum", [b.Mul(centered, centered)]), inverse));
        var cutoff = b.Scale(std, (float)NormalDistribution.Quantile(sparsity));
        return b.Unary("Relu", b.Sub(centered, cutoff), $"{p}sparse_gate");
    }

    // The activation of the dense MLP and the shared expert of layer il: SwiGLU clamped to mlp.swiglu_clamp when the layer has a limit.
    private IMlpActivation DenseActivationOf(int il)
        => il < _hp.NLayer && _hp.SwigluClampOf(il) is > 0f and var limit ? new ClampedSwigluActivation(limit, _hp.SwigluClampInput) : _activation;

    // The activation of the routed experts of layer il: SwiGLU clamped to moe.swiglu_clamp when the layer has a limit.
    private IMlpActivation ExpertActivationOf(int il)
        => il < _hp.NLayer && _hp.MoeSwigluClampOf(il) is > 0f and var limit ? new ClampedSwigluActivation(limit, _hp.MoeSwigluClampInput) : _activation;

    // BitNet sub-layer norm: an RMS norm of the input of the output projection (attention or MLP) when the layer has its weight.
    private Value SubNorm(GraphBuilder b, Value x, string weightName)
        => b.TryWeight(weightName) is { } weight ? b.RmsNorm(x, weight, _hp.Eps) : x;
}
