using Sharplight.Graph;

namespace Sharplight.Architectures;

/// <summary>
/// Hierarchical layer cycles (HRM-Text): a low and a high stack of layers run alternately over the same tokens, with weights shared across
/// passes and KV caches of their own per pass. With the embeddings as the high state z_H and the learned <c>cycles.low_init</c> [H] as the
/// low state z_L, every high cycle runs z_L = low(z_L + z_H) L times and then z_H = high(z_H + z_L); the trunk's output is the final z_H.
/// Each stack pass ends with its own norm (<see cref="DecoderNorm"/> of <c>cycles.low_norm.weight</c> / <c>cycles.high_norm.weight</c>,
/// weight-less when absent).
/// </summary>
internal static class HierarchicalCycles
{
    /// <summary>Weight of the initial low state, [H].</summary>
    public const string LowInitWeight = "cycles.low_init";

    /// <summary>
    /// Emits the cycles. <paramref name="stack"/> runs one pass: the input, the first cache slot of the pass and the first physical layer of
    /// its stack; it returns the output of the pass's last layer (before the stack norm).
    /// </summary>
    public static Value Build(GraphBuilder b, Value embeddings, HierarchicalCycleOptions o, DecoderNorm norm, float eps, Func<Value, int, int, Value> stack)
    {
        var high = embeddings;
        var low = b.Weight(LowInitWeight);
        for (int h = 0; h < o.HighCycles; h++)
        {
            for (int l = 0; l <= o.LowCycles; l++)
            {
                bool highPass = l == o.LowCycles;
                int slotBase = (h * (o.LowCycles + 1) + l) * o.StackLayers;
                string name = highPass ? "high" : "low";
                using var _ = b.Group($"cycles.{h}.{name}{(highPass ? "" : l)}");
                var input = b.Add(high, low, $"cycles.{h}.{l}.input");
                var output = norm.Apply(b, stack(input, slotBase, highPass ? o.StackLayers : 0), $"cycles.{name}_norm.weight", eps, $"cycles.{h}.{l}.norm");
                if (highPass)
                {
                    high = output;
                }
                else
                {
                    low = output;
                }
            }
        }

        return high;
    }
}
