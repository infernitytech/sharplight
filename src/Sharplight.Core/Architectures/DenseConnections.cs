using Sharplight.Graph;

namespace Sharplight.Architectures;

/// <summary>
/// Dynamic dense cross-layer connections (MUDD, Limite): a layer reads, instead of the previous layer's output alone, per-token mixtures of the
/// inputs of earlier layers (its taps, the layer's own input among them). The mixing weights come from the layer input through a small MLP,
/// w = up(gelu(down(rms(x)))) + up.bias with n · S outputs (n taps, S mixtures); mixture s is Σ_j w[s·n + j] · history[tap j]. Weights:
/// <c>{p}dense.down.weight</c> [F, H], <c>{p}dense.up.weight</c> [S·n, F], <c>{p}dense.up.bias</c> [S·n].
/// </summary>
internal static class DenseConnections
{
    /// <summary>Whether the layer with prefix <paramref name="p"/> has dense connections.</summary>
    public static bool IsPresent(GraphBuilder b, string p) => b.HasWeight(p + "dense.up.weight");

    /// <summary>
    /// Emits the mixtures of one layer. <paramref name="history"/> holds the inputs of layers 0..il ([T, H] each), <paramref name="x"/> is the
    /// layer's own input (history[il]); returns <paramref name="mixtures"/> values [T, H] in the order of the up projection's row blocks.
    /// </summary>
    public static Value[] Mix(GraphBuilder b, IReadOnlyList<Value> history, Value x, int[] taps, string p, float eps, int mixtures)
    {
        var hidden = b.Unary("Gelu", b.Linear(b.RmsNorm(x, null, eps), p + "dense.down.weight"));
        var w = b.Add(b.Linear(hidden, p + "dense.up.weight"), b.Weight(p + "dense.up.bias"), $"{p}dense_weights");
        var result = new Value[mixtures];
        for (int s = 0; s < mixtures; s++)
        {
            Value? sum = null;
            for (int j = 0; j < taps.Length; j++)
            {
                var term = b.Mul(history[taps[j]], b.Slice(w, 1, (long)s * taps.Length + j, 1));
                sum = sum is null ? term : b.Add(sum, term);
            }

            result[s] = sum!;
        }

        return result;
    }
}
