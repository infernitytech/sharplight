using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Hugging Face <c>spark2_5</c> (Spark 2.5 / X2.6, remote code): RMS norms, a fused <c>q_k_v_proj</c> (query rows, then key rows, then value rows),
/// sliding-window layers from <c>layer_types</c> on their own rotary base and width (<c>rope_parameters.sliding_attention</c>), partial rotation of the
/// full layers, a per-head sigmoid output gate of the attention (<c>g_proj</c>, [heads, hidden]) and a GELU-gated MLP → canonical <c>decoder</c>.
/// </summary>
internal sealed class Spark25HfMapping : HfDecoderFamilyMapping
{
    private const string SlidingAttention = "sliding_attention";

    private static readonly IReadOnlyDictionary<string, string> s_layerTensors = With(LlamaLayerTensors,
        ("self_attn.g_proj.weight", "attention.gate.weight"),
        ("self_attn.out_proj.weight", "attention.o.weight"));

    private static readonly IReadOnlyDictionary<string, string> s_globalTensors = With(LlamaGlobalTensors,
        ("model.embedding.weight", "token_embedding.weight"));

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["spark2_5"];

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> LayerTensors => s_layerTensors;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> GlobalTensors => s_globalTensors;

    /// <inheritdoc/>
    public override IEnumerable<string> Validate(HfConfig config)
    {
        if (config.GetBool("headwise_attn_output_gate") == false || (config.GetString("gate_attn_act_mode") ?? "sigmoid") != "sigmoid")
        {
            yield return "spark2_5: only the per-head sigmoid output gate of the attention is supported.";
        }
    }

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        if (AddSlidingLayers(md, c) is null)
        {
            return;
        }

        md["decoder.rope.local_base"] = (float)(c.LayerTypeRopeDouble(SlidingAttention, "rope_theta") ?? c.RopeDouble("rope_theta", "rope_theta") ?? 10000.0);
        md["decoder.rope.local_dim"] = (int)(HeadDimOf(c) * (c.LayerTypeRopeDouble(SlidingAttention, "partial_rotary_factor") ?? 1.0));
    }

    /// <inheritdoc/>
    protected override bool TryMapSpecialLayerTensor(HfConfig c, string layerPrefix, string suffix, WeightInfo w, HfWeightCollector collector)
    {
        if (suffix != "self_attn.q_k_v_proj.weight")
        {
            return false;
        }

        int headDim = HeadDimOf(c);
        int queryRows = HeadCount(c) * headDim;
        int kvRows = KvHeadCount(c, HeadCount(c)) * headDim;
        collector.Add(w.RowSlice(layerPrefix + "attention.q.weight", 0, queryRows));
        collector.Add(w.RowSlice(layerPrefix + "attention.k.weight", queryRows, kvRows));
        collector.Add(w.RowSlice(layerPrefix + "attention.v.weight", queryRows + kvRows, kvRows));
        return true;
    }
}
