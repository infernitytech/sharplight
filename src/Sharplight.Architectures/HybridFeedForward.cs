using Sharplight.Graph;
using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// Gated feed-forward blocks of the <see cref="HybridAdapter"/>: a dense SwiGLU-style MLP (<c>mlp.{gate,up,down}.weight</c>) and/or a routed
/// mixture of such MLPs (<c>mlp.router.weight</c> [E, hidden], rank-3 <c>mlp.experts.{gate,up,down}.weight</c> [E, I, hidden] / [E, hidden, I]);
/// in a routed layer the dense MLP is the shared expert and lives under <c>mlp.shared.*</c>. A layer may have both; their outputs are summed.
/// </summary>
internal sealed class HybridFeedForward
{
    private readonly string _activation;
    private readonly MoeOptions _moe;
    private readonly int _hidden;

    /// <summary>Reads <c>mlp.activation</c> and the <c>moe.*</c> hyper-parameters (<c>expert_count</c>, <c>expert_used_count</c>, <c>normalize_weights</c>, <c>router_activation</c>, <c>weights_scale</c>).</summary>
    public HybridFeedForward(ModelSource src, int hidden)
    {
        _hidden = hidden;
        _activation = src.Arch("mlp.activation", "silu");
        _moe = MoeOptions.Read(src, normalizeDefault: true);
    }

    /// <summary>Whether the layer with weight prefix <paramref name="p"/> has any feed-forward weights.</summary>
    public static bool Exists(ModelSource src, string p)
        => src.Weight(p + "mlp.up.weight") is not null || src.Weight(p + "mlp.router.weight") is not null;

    /// <summary>Emits the feed-forward of one layer on the normalized input <paramref name="h"/>.</summary>
    public Value Build(GraphBuilder b, Value h, string p)
    {
        if (!b.HasWeight(p + "mlp.router.weight"))
        {
            return Dense(b, h, p + "mlp.", p);
        }

        return MoeBlock.Build(b, h, p + "mlp.", _moe, _hidden, t => Act(b, t));
    }

    private Value Act(GraphBuilder b, Value t) => _activation switch
    {
        "silu" => b.Silu(t),
        "gelu" => b.Unary("Gelu", t),
        "relu" => b.Unary("Relu", t),
        _ => throw new NotSupportedException($"hybrid: unknown mlp.activation '{_activation}'."),
    };

    private Value Dense(GraphBuilder b, Value h, string prefix, string layerPrefix)
    {
        var act = b.Mul(Act(b, b.Linear(h, prefix + "gate.weight")), b.Linear(h, prefix + "up.weight"));
        return b.Linear(act, prefix + "down.weight", name: $"{layerPrefix}ffn_out");
    }
}
