namespace Sharplight.Import.HuggingFace;

internal sealed partial class Lfm2HfMapping
{
    /// <summary>The <c>layer_types</c> entry of LFM3's sliding-window attention layers.</summary>
    private const string SlidingAttention = "sliding_attention";

    /// <summary>
    /// LFM3's attention layers: <c>layer_types</c> marks the <c>sliding_attention</c> layers (window <c>sliding_window</c>, the rotary base of their own
    /// <c>rope_parameters</c> entry); the <c>full_attention</c> layers rotate with the global base.
    /// </summary>
    private static void AddLfm3HyperParameters(Dictionary<string, object> md, string a, HfConfig c)
    {
        if (c.LayerTypeRopeDouble(SlidingAttention, "partial_rotary_factor") is { } slidingPartial && slidingPartial != 1.0)
        {
            throw new NotSupportedException("lfm3: partial rotation of the sliding-window layers is not supported.");
        }

        if (c.GetStringArray("layer_types") is not { } layerTypes || !layerTypes.Contains(SlidingAttention) || (c.GetInt("sliding_window") ?? 0) <= 0)
        {
            return;
        }

        md[a + "attention.sliding_window"] = c.Int("sliding_window");
        md[a + "attention.sliding_layers"] = HfMappingHelpers.SlidingFlags(layerTypes);
        md[a + "rope.local_base"] = (float)(c.LayerTypeRopeDouble(SlidingAttention, "rope_theta") ?? c.RopeDouble("rope_theta", "rope_theta") ?? 10000.0);
    }

    /// <summary>
    /// LFM3 rotates only the first <c>partial_rotary_factor · head_dim / 2</c> frequency pairs of its full-attention layers, at the frequencies of the
    /// whole head ("zero-tail" partial rotation): the remaining pairs get a frequency factor so large that their angle is zero.
    /// </summary>
    private static void AddLfm3FrequencyFactors(HfConfig c, HfWeightCollector collector)
    {
        double partial = c.LayerTypeRopeDouble("full_attention", "partial_rotary_factor") ?? c.GetDouble("partial_rotary_factor") ?? 1.0;
        if (partial >= 1.0)
        {
            return;
        }

        int headDim = c.GetInt("head_dim") ?? c.Int("hidden_size") / c.Int("num_attention_heads");
        int rotated = (int)(partial * headDim) / 2;
        var factors = new float[headDim / 2];
        for (int i = 0; i < factors.Length; i++)
        {
            factors[i] = i < rotated ? 1f : 1e30f;
        }

        collector.Add(HfMappingHelpers.F32Weight("rope.freq_factors", factors));
    }
}
