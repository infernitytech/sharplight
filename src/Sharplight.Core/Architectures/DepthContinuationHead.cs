using Sharplight.Graph;

namespace Sharplight.Architectures;

/// <summary>
/// The continuation head at the end of a pass of a multi-pass stack: a small MLP over the pass's state, its change and the retrieved memory that
/// estimates per token whether another pass is worthwhile, as a probability in (0, 1). The value is a signal for later layers (it is written into
/// the <see cref="DepthMemory"/>); every pass still runs.
/// </summary>
/// <remarks>
/// Weights below <c>continuation.</c>: <c>state.weight</c>, <c>retrieved.weight</c>, <c>diff.weight</c> [R, H], <c>route.weight</c> [R, R],
/// <c>bias</c> [R], <c>out.weight</c> [1, R], <c>out.bias</c> [1].
/// h = silu(W_s·state + W_r·retrieved + W_d·diff + W_f·features + b); logit = W_o·h + b_o + tanh(rms(diff) / max(rms(state), 1e-6)) − ½;
/// result = sigmoid(clamp(logit, ±limit)).
/// </remarks>
internal static class DepthContinuationHead
{
    private const string Prefix = "continuation.";

    /// <summary>Emits the head on the pass state, retrieved summary and change [T, H] and the routing features [T, R]; returns [T, 1].</summary>
    public static Value Build(GraphBuilder b, Value state, Value retrieved, Value diff, Value features, int hidden, float limit)
    {
        var h = b.Linear(state, Prefix + "state.weight");
        h = b.Add(h, b.Linear(retrieved, Prefix + "retrieved.weight"));
        h = b.Add(h, b.Linear(diff, Prefix + "diff.weight"));
        h = b.Add(h, b.Linear(features, Prefix + "route.weight"));
        h = b.Silu(b.Add(h, b.Weight(Prefix + "bias")));
        var logit = b.Linear(h, Prefix + "out.weight", Prefix + "out.bias");                                                    // [T, 1]
        var scale = b.Emit1("Clip", [Rms(b, state, hidden)], Attributes.From(("min", 1e-6f), ("max", float.MaxValue)));
        var innovation = b.Unary("Tanh", b.Binary("Div", Rms(b, diff, hidden), scale));
        logit = b.Add(logit, b.Scale(innovation, 1f, -0.5f));
        return b.Sigmoid(b.Emit1("Clip", [logit], Attributes.From(("min", -limit), ("max", limit))), "continuation");
    }

    // sqrt(mean(x²)) over the last axis: [T, 1].
    private static Value Rms(GraphBuilder b, Value x, int hidden)
        => b.Unary("Sqrt", b.Scale(b.Unary("ReduceSum", b.Mul(x, x)), 1f / hidden));
}
