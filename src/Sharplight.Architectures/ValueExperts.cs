using Sharplight.Graph;

namespace Sharplight.Architectures;

/// <summary>
/// Mixture of value experts (MoVA, K2-Horizon): the attention values of a row are Σ_k w_k·silu(W_k·h) over the k value experts its router picks,
/// in place of one value projection. Routing goes through <see cref="MoeRouter"/> (options <see cref="ValueExpertOptions"/>), the projections
/// through MatMulId.
/// </summary>
/// <remarks>
/// Canonical weights below <c>prefix</c> (<c>layers.{i}.attention.value_experts.</c>): <c>weight</c> [E, N, H] (N = kv heads · head size),
/// <c>router.weight</c> [E, H] and the optional selection bias <c>router_bias</c> [E].
/// </remarks>
internal static class ValueExperts
{
    /// <summary>Returns <c>true</c> when the layer at <paramref name="prefix"/> has value experts.</summary>
    public static bool IsPresent(GraphBuilder b, string prefix) => b.HasWeight(prefix + "weight");

    /// <summary>Emits the routed values of the rows <paramref name="h"/> [T, H] and returns them [T, N].</summary>
    /// <param name="b">The graph builder.</param>
    /// <param name="h">The normed layer input.</param>
    /// <param name="prefix">Weight-name prefix of the block, ending in a dot.</param>
    /// <param name="options">The routing.</param>
    public static Value Build(GraphBuilder b, Value h, string prefix, ValueExpertOptions options)
    {
        int k = options.Routing.ExpertsUsed;
        var (weights, ids) = MoeRouter.Route(b, h, prefix, options.Routing);
        var projected = b.Emit1("MatMulId", [h, b.Weight(prefix + "weight"), ids], null, $"{prefix}projected");                 // [T, k, N]
        var weighted = b.Mul(b.Silu(projected), b.Reshape(weights, 0, k, 1));
        var summed = b.Emit1("ReduceSum", [b.Permute(weighted, 0, 2, 1)]);                                                       // [T, N, 1]
        return b.Reshape(summed, 0, projected.Shape[-1].Fixed);
    }
}
