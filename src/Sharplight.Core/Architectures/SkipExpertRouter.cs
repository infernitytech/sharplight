using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// The MLP router of ZAYA1: a projection to a small router width, a router state carried from layer to layer, an RMS norm, two GELU (erf) layers
/// and logits over the E experts plus one skip class. The top-1 expert is chosen among the real experts by the biased probabilities; when the
/// skip class scores higher than it, the block's output is zero.
/// </summary>
/// <remarks>
/// Weights below the block prefix (e.g. <c>layers.3.mlp.</c>): <c>router.down.weight</c> [R, H] and <c>router.down.bias</c> [R];
/// <c>router.state_scale</c> [R] (absent on the first layer): the router rows add the previous layer's router rows times this scale, and the
/// sum is the state handed to the next layer; <c>router.norm.weight</c> [R]; <c>router.fc1.{weight,bias}</c>, <c>router.fc2.{weight,bias}</c>
/// [R, R]; <c>router.out.weight</c> [E + 1, R] (the last row scores the skip class); <c>router.class_bias</c> [E + 1], added to the softmax
/// probabilities for the selection only. The mixing weight of the chosen expert is its unbiased probability, zero when the skip class wins.
/// </remarks>
internal static class SkipExpertRouter
{
    /// <summary>Whether the block at <paramref name="prefix"/> (e.g. <c>layers.3.mlp.</c>) routes with this router.</summary>
    public static bool IsPresent(GraphBuilder b, string prefix) => b.HasWeight(prefix + "router.out.weight");

    /// <summary>Routes the rows <paramref name="h"/> [T, H] to one expert each.</summary>
    /// <param name="b">The graph builder.</param>
    /// <param name="h">The normalized rows [T, H].</param>
    /// <param name="prefix">Weight-name prefix of the block.</param>
    /// <param name="experts">The number of real experts E.</param>
    /// <param name="eps">Epsilon of the router norm.</param>
    /// <param name="state">The previous layer's router rows [T, R] (null on the first routed layer); replaced by this layer's.</param>
    /// <returns>Mixing weights f32 [T, 1] and expert indices i32 [T, 1].</returns>
    public static (Value Weights, Value Ids) Route(GraphBuilder b, Value h, string prefix, int experts, float eps, ref Value? state)
    {
        string r = prefix + "router.";
        var rows = b.Linear(h, r + "down.weight", r + "down.bias");
        if (state is not null && b.TryWeight(r + "state_scale") is { } stateScale)
        {
            rows = b.Add(rows, b.Mul(state, stateScale), $"{r}state");
        }

        state = rows;
        var x = b.RmsNorm(rows, b.Weight(r + "norm.weight"), eps);
        x = b.Unary("Gelu", b.Linear(x, r + "fc1.weight", r + "fc1.bias"));
        x = b.Unary("Gelu", b.Linear(x, r + "fc2.weight", r + "fc2.bias"));
        var probs = b.Emit1("Softmax", [b.Linear(x, r + "out.weight")], null, $"{prefix}router_probs");                 // [T, E + 1]
        var scores = b.TryWeight(r + "class_bias") is { } classBias ? b.Add(probs, classBias) : probs;
        var top = b.Emit("TopK", [b.Slice(scores, 1, 0, experts)], Attributes.From(("k", 1L)), $"{prefix}router_topk");
        var keep = b.Binary("LessEqual", b.Slice(scores, 1, experts, 1), top[0], $"{prefix}router_keep");            // [T, 1]

        var iotaBytes = new byte[experts * 4];
        for (int e = 0; e < experts; e++)
        {
            BitConverter.TryWriteBytes(iotaBytes.AsSpan(e * 4, 4), (float)e);
        }

        var iota = b.Constant($"{prefix}router_iota", DataType.F32, new Shape(experts), iotaBytes);
        var onehot = b.Binary("Equal", b.Emit1("Cast", [top[1]], Attributes.From(("type", "f32"))), iota);            // [T, E]
        var picked = b.Emit1("ReduceSum", [b.Mul(onehot, b.Slice(probs, 1, 0, experts))]);                             // [T, 1]
        return (b.Mul(picked, keep, $"{prefix}router_weights"), top[1]);
    }
}
