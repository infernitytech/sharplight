using Sharplight.Graph;
using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// Gated feed-forward blocks of the <see cref="HybridAdapter"/>: a dense SwiGLU-style MLP (<c>mlp.{gate,up,down}.weight</c>) and/or a routed
/// mixture of such MLPs (<c>mlp.router.weight</c> [E, hidden], rank-3 <c>mlp.experts.{gate,up,down}.weight</c> [E, I, hidden] / [E, hidden, I]);
/// in a routed layer the dense MLP is the shared expert and lives under <c>mlp.shared.*</c>. A layer may have both; their outputs are summed.
/// A layer with <c>mlp.router.out.weight</c> routes with the <see cref="SkipExpertRouter"/> (ZAYA1) instead of a linear router.
/// </summary>
internal sealed class HybridFeedForward
{
    private readonly IMlpActivation _activation;
    private readonly MoeOptions _moe;
    private readonly int _hidden;
    private readonly float _eps;

    /// <summary>Reads <c>mlp.activation</c> and the <c>moe.*</c> hyper-parameters (<c>expert_count</c>, <c>expert_used_count</c>, <c>normalize_weights</c>, <c>router_activation</c>, <c>weights_scale</c>).</summary>
    public HybridFeedForward(ModelSource src, int hidden)
    {
        _hidden = hidden;
        _eps = src.Arch("norm_eps", 1e-5f);
        _activation = MlpActivations.Create(src.Arch("mlp.activation", "silu"));
        _moe = MoeOptions.Read(src, normalizeDefault: true);
    }

    /// <summary>Whether the layer with weight prefix <paramref name="p"/> has any feed-forward weights.</summary>
    public static bool Exists(ModelSource src, string p)
        => src.Weight(p + "mlp.up.weight") is not null || src.Weight(p + "mlp.router.weight") is not null || src.Weight(p + "mlp.router.out.weight") is not null;

    /// <summary>Emits the feed-forward of one layer on the normalized input <paramref name="h"/>.</summary>
    /// <param name="b">The graph builder.</param>
    /// <param name="h">The normalized input [T, hidden].</param>
    /// <param name="p">The layer's weight-name prefix.</param>
    /// <param name="routerState">The router rows the previous <see cref="SkipExpertRouter"/> layer handed on (null before the first); replaced by
    /// this layer's when it routes that way.</param>
    public Value Build(GraphBuilder b, Value h, string p, ref Value? routerState)
    {
        if (SkipExpertRouter.IsPresent(b, p + "mlp."))
        {
            var route = SkipExpertRouter.Route(b, h, p + "mlp.", _moe.Experts, _eps, ref routerState);
            return MoeBlock.Build(b, h, p, "mlp.", _moe, _hidden, _activation, route: route);
        }

        if (!b.HasWeight(p + "mlp.router.weight"))
        {
            return Dense(b, h, p + "mlp.", p);
        }

        return MoeBlock.Build(b, h, p, "mlp.", _moe, _hidden, _activation);
    }

    private Value Dense(GraphBuilder b, Value h, string prefix, string layerPrefix)
    {
        var act = _activation.ActivateGated(b, b.Linear(h, prefix + "gate.weight"), b.Linear(h, prefix + "up.weight"), layerPrefix);
        return b.Linear(act, prefix + "down.weight", name: $"{layerPrefix}ffn_out");
    }
}
