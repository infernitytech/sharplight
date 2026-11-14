using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>granite</c> and <c>granitemoe</c> (IBM Granite 3.x): the Llama decoder with muP multipliers — <c>embedding_multiplier</c>,
/// <c>residual_multiplier</c> (every sub-layer output), <c>attention_multiplier</c> (replaces 1/sqrt(head_dim)) and <c>logits_scaling</c> (a divisor).
/// The MoE variant stores its experts fused: <c>input_linear</c> [E, 2·ff, H] (gate first, then up), <c>output_linear</c> [E, H, ff] and a <c>router.layer</c>.
/// </summary>
internal sealed class GraniteHfMapping : HfDecoderFamilyMapping
{
    private static readonly IReadOnlyDictionary<string, string> s_layerTensors = With(LlamaLayerTensors,
        ("block_sparse_moe.router.layer.weight", "mlp.router.weight"),
        ("block_sparse_moe.output_linear.weight", "mlp.experts.down.weight"));

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["granite", "granitemoe"];

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> LayerTensors => s_layerTensors;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        const string prefix = "decoder.";
        AddMupScales(md, c);
        if (c.ModelType == "granitemoe")
        {
            md[prefix + "moe.expert_count"] = c.Int("num_local_experts");
            md[prefix + "moe.expert_used_count"] = c.Int("num_experts_per_tok");
            md[prefix + "moe.ffn_hidden_size"] = c.Int("intermediate_size");
            md[prefix + "moe.normalize_weights"] = true;
        }
    }

    /// <summary>
    /// Granite's muP multipliers as canonical decoder keys: <c>attention_multiplier</c> (attention scale), <c>embedding_multiplier</c>,
    /// <c>residual_multiplier</c> (every sub-layer output) and <c>logits_scaling</c> (a divisor of the logits).
    /// </summary>
    internal static void AddMupScales(Dictionary<string, object> md, HfConfig c)
    {
        const string prefix = "decoder.";
        if (c.GetDouble("attention_multiplier") is { } attention)
        {
            md[prefix + "attention.scale"] = (float)attention;
        }

        if (c.GetDouble("embedding_multiplier") is { } embedding)
        {
            md[prefix + "embedding_scale"] = (float)embedding;
        }

        if (c.GetDouble("residual_multiplier") is { } residual)
        {
            md[prefix + "residual.scale"] = (float)residual;
        }

        if (c.GetDouble("logits_scaling") is { } divisor && divisor != 0)
        {
            md[prefix + "logit_scale"] = (float)(1.0 / divisor);
        }
    }

    /// <inheritdoc/>
    protected override bool TryMapSpecialLayerTensor(HfConfig c, string layerPrefix, string suffix, WeightInfo w, HfWeightCollector collector)
    {
        if (suffix != "block_sparse_moe.input_linear.weight")
        {
            return false;
        }

        var (gate, up) = HfFusedProjections.SplitSecondAxis(w, layerPrefix + "mlp.experts.gate.weight", layerPrefix + "mlp.experts.up.weight", c.Int("intermediate_size"));
        collector.Add(gate);
        collector.Add(up);
        return true;
    }
}
