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
    private readonly int _experts;
    private readonly int _expertsUsed;
    private readonly bool _normalizeWeights;
    private readonly bool _sigmoidRouter;
    private readonly float _weightsScale;
    private readonly int _hidden;

    /// <summary>Reads <c>mlp.activation</c> and the <c>moe.*</c> hyper-parameters (<c>expert_count</c>, <c>expert_used_count</c>, <c>normalize_weights</c>, <c>router_activation</c>, <c>weights_scale</c>).</summary>
    public HybridFeedForward(ModelSource src, int hidden)
    {
        _hidden = hidden;
        _activation = src.Arch("mlp.activation", "silu");
        _experts = src.Arch("moe.expert_count", 0);
        _expertsUsed = src.Arch("moe.expert_used_count", 0);
        _normalizeWeights = src.Arch("moe.normalize_weights", true);
        _sigmoidRouter = src.Arch("moe.router_activation", "softmax") == "sigmoid";
        _weightsScale = src.Arch("moe.weights_scale", 1f);
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

        var routed = Routed(b, h, p);
        return b.HasWeight(p + "mlp.shared.up.weight") ? b.Add(routed, Dense(b, h, p + "mlp.shared.", p)) : routed;
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

    private Value Routed(GraphBuilder b, Value h, string p)
    {
        if (_experts <= 0 || _expertsUsed <= 0)
        {
            throw new InvalidDataException("hybrid: expert weights present but moe.expert_count / moe.expert_used_count are missing.");
        }

        var logits = b.Linear(h, p + "mlp.router.weight");
        var probs = _sigmoidRouter ? b.Sigmoid(logits, $"{p}router_probs") : b.Emit1("Softmax", [logits], null, $"{p}router_probs");
        var top = b.Emit("TopK", [probs], Attributes.From(("k", (long)_expertsUsed)), $"{p}router_topk");
        var weights = top[0];
        var ids = top[1];
        if (_normalizeWeights)
        {
            weights = b.Binary("Div", weights, b.Emit1("ReduceSum", [weights]));
        }

        if (_weightsScale != 1f)
        {
            weights = b.Scale(weights, _weightsScale);
        }

        var up = b.Emit1("MatMulId", [h, b.Weight(p + "mlp.experts.up.weight"), ids], null, $"{p}experts_up");
        var gate = b.Emit1("MatMulId", [h, b.Weight(p + "mlp.experts.gate.weight"), ids], null, $"{p}experts_gate");
        var down = b.Emit1("MatMulId", [b.Mul(Act(b, gate), up), b.Weight(p + "mlp.experts.down.weight"), ids], null, $"{p}experts_down");   // [T, k, H]
        var weighted = b.Mul(down, b.Reshape(weights, 0, _expertsUsed, 1));
        var summed = b.Emit1("ReduceSum", [b.Permute(weighted, 0, 2, 1)]);                                                                        // [T, H, 1]
        return b.Reshape(summed, 0, _hidden);
    }
}
