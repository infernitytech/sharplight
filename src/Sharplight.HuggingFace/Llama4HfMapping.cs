using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>llama4</c> / <c>llama4_text</c> (Llama 4 Scout / Maverick; the text model of the multimodal checkpoints, the
/// vision tower is not imported): interleaved rotary pairs on the layers flagged in <c>no_rope_layers</c> (1 = rotary), chunked
/// attention (<c>attention_chunk_size</c>) and an unweighted q/k RMS norm after the rotary embedding on those layers; the NoPE layers
/// attend globally with position-dependent query temperature (<c>attn_scale</c>, <c>floor_scale</c>). MoE layers route each token
/// to its top-k experts with sigmoid gates that scale the expert inputs and add a shared expert; the fused experts are stored
/// [E, in, out] (<c>gate_up_proj</c> gate columns first).
/// </summary>
internal sealed partial class Llama4HfMapping : HfMoeFamilyMapping
{
    private const float TemperatureOffset = 1f;

    private static readonly IReadOnlyDictionary<string, string> s_layerTensors = With(LlamaLayerTensors,
        ("feed_forward.gate_proj.weight", "mlp.gate.weight"),
        ("feed_forward.up_proj.weight", "mlp.up.weight"),
        ("feed_forward.down_proj.weight", "mlp.down.weight"),
        ("feed_forward.router.weight", "mlp.router.weight"),
        ("feed_forward.shared_expert.gate_proj.weight", "mlp.shared.gate.weight"),
        ("feed_forward.shared_expert.up_proj.weight", "mlp.shared.up.weight"),
        ("feed_forward.shared_expert.down_proj.weight", "mlp.shared.down.weight"));

    private static readonly IReadOnlyDictionary<string, string> s_globalTensors = new Dictionary<string, string>(LlamaGlobalTensors.Concat(
    [
        new("language_model.model.embed_tokens.weight", "token_embedding.weight"),
        new("language_model.model.norm.weight", "output_norm.weight"),
        new("language_model.lm_head.weight", "lm_head.weight"),
    ]), StringComparer.Ordinal);

    [GeneratedRegex(@"^(?:language_model\.)?model\.layers\.(\d+)\.(.+)$")]
    private static partial Regex Llama4LayerName();

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["llama4", "llama4_text"];

    /// <inheritdoc/>
    protected override string RopeStyle => "gptj";

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> LayerTensors => s_layerTensors;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> GlobalTensors => s_globalTensors;

    /// <inheritdoc/>
    protected override Regex LayerName => Llama4LayerName();

    /// <inheritdoc/>
    protected override int FeedForwardSize(HfConfig c) => c.GetInt("intermediate_size_mlp") ?? c.Int("intermediate_size");

    /// <inheritdoc/>
    protected override bool IsIgnored(string name)
        => base.IsIgnored(name) || name.StartsWith("vision_model.", StringComparison.Ordinal) || name.StartsWith("multi_modal_projector.", StringComparison.Ordinal);

    /// <inheritdoc/>
    protected override MoeRouting Routing(HfConfig c)
        => new(c.Int("num_local_experts"), c.Int("num_experts_per_tok"), c.Int("intermediate_size"), "sigmoid", Normalize: false, WeightsBeforeExperts: true);

    /// <inheritdoc/>
    protected override void AddFamilyHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        const string prefix = "decoder.";
        int layerCount = LayerCount(c);
        int[] rope = IntArray(c, "no_rope_layers") is { Length: > 0 } flags && flags.Length == layerCount ? flags : [.. Enumerable.Repeat(1, layerCount)];
        int[] noRope = [.. rope.Select(flag => flag == 0 ? 1 : 0)];
        md[prefix + "rope.layers"] = rope;
        if (c.GetInt("attention_chunk_size") is > 0 and var chunk)
        {
            md[prefix + "attention.chunk_size"] = chunk;
            md[prefix + "attention.chunk_layers"] = rope;
        }

        if (c.GetBool("use_qk_norm") == true)
        {
            md[prefix + "attention.qk_norm_unweighted"] = true;
            md[prefix + "attention.qk_norm_layers"] = rope;
            md[prefix + "attention.qk_norm_after_rope"] = true;
        }

        if (c.Get("attn_temperature_tuning") is { } tuning && tuning.ValueKind is not (System.Text.Json.JsonValueKind.False or System.Text.Json.JsonValueKind.Null)
            && !(tuning.ValueKind == System.Text.Json.JsonValueKind.Number && tuning.GetDouble() == 0))
        {
            md[prefix + "attention.temperature_scale"] = (float)c.Double("attn_scale");
            md[prefix + "attention.temperature_length"] = (float)c.Double("floor_scale");
            md[prefix + "attention.temperature_offset"] = TemperatureOffset;
            md[prefix + "attention.temperature_layers"] = noRope;
        }
    }

    /// <inheritdoc/>
    protected override bool TryMapFamilyTensor(HfConfig c, string layerPrefix, string suffix, WeightInfo w, HfWeightCollector collector)
    {
        switch (suffix)
        {
            case "feed_forward.experts.gate_up_proj":
                collector.Add(HfExpertLayouts.Transposed(w, layerPrefix + "mlp.experts.gate_up.weight"));
                return true;
            case "feed_forward.experts.down_proj":
                collector.Add(HfExpertLayouts.Transposed(w, layerPrefix + "mlp.experts.down.weight"));
                return true;
            default:
                return false;
        }
    }
}
