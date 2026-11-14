using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// Qwen3-VL (HF <c>qwen3_vl</c>, <c>qwen3_vl_text</c>) and Qwen3-VL-MoE (<c>qwen3_vl_moe</c>, <c>qwen3_vl_moe_text</c>) → canonical <c>decoder</c>:
/// the language model is Qwen3 (or Qwen3-MoE) with interleaved M-RoPE (<c>rope.sections</c> from <c>mrope_section</c>, padded to 4 axes) and the
/// vision tower (with its DeepStack mergers) is mapped by <see cref="QwenVisionHfMapping"/> into the same source. Qwen3-VL-MoE stores every
/// layer's experts fused: <c>mlp.experts.gate_up_proj</c> [E, H, 2I] (gate columns first) and <c>mlp.experts.down_proj</c> [E, I, H], or already
/// in the [E, out, in] order transformers keeps in memory (the order is told by the shapes; when 2I = H the stored [E, in, out] is assumed).
/// </summary>
internal sealed partial class Qwen3VlHfMapping : IHfArchitectureMapping
{
    private const string LanguageModelPrefix = "model.language_model.";
    private readonly DecoderHfMapping _text = new();

    /// <inheritdoc/>
    public string Architecture => _text.Architecture;

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["qwen3_vl", "qwen3_vl_text", "qwen3_vl_moe", "qwen3_vl_moe_text"];

    [GeneratedRegex(@"^model\.(?:language_model\.)?layers\.(\d+)\.mlp\.(experts\.gate_up_proj|experts\.down_proj|gate\.weight)$")]
    private static partial Regex FusedExperts();

    // transformers 5 writes num_local_experts.
    private static int? ExpertCount(HfConfig c) => c.GetInt("num_experts") ?? c.GetInt("num_local_experts");

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig config, HfImportOptions options)
    {
        var md = new Dictionary<string, object>(_text.MapHyperParameters(config, options), StringComparer.Ordinal);
        if (config.RopeIntArray("mrope_section") is { } sections)
        {
            var axes = new int[4];
            for (int i = 0; i < Math.Min(axes.Length, sections.Length); i++)
            {
                axes[i] = sections[i];
            }

            md[Architecture + ".rope.sections"] = axes;
            md[Architecture + ".rope.sections_interleaved"] = config.GlobalRopeParameters is { } rp && rp.TryGetProperty("mrope_interleaved", out var il)
                ? il.ValueKind == System.Text.Json.JsonValueKind.True : true;
        }

        if (ExpertCount(config) is { } experts)
        {
            if (config.GetInt("decoder_sparse_step") is { } step && step != 1 || config.Get("mlp_only_layers") is { ValueKind: System.Text.Json.JsonValueKind.Array } dense && dense.GetArrayLength() > 0)
            {
                throw new NotSupportedException($"{config.ModelType}: dense layers between MoE layers (decoder_sparse_step, mlp_only_layers) are not supported.");
            }

            string a = Architecture + ".";
            md[a + "moe.expert_count"] = experts;
            md[a + "moe.expert_used_count"] = config.Int("num_experts_per_tok");
            md[a + "moe.ffn_hidden_size"] = config.Int("moe_intermediate_size");
            md[a + "moe.normalize_weights"] = config.GetBool("norm_topk_prob") ?? true;
        }

        QwenVisionHfMapping.AddVisionHyperParameters(config, md, Architecture);
        return md;
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig config, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HfImportOptions options)
    {
        // model.language_model.X → model.X, the layout of the plain text model; the tower is mapped separately.
        var text = tensors
            .Where(t => !QwenVisionHfMapping.IsVisionTensor(t.Key) && !(ExpertCount(config) is not null && FusedExperts().IsMatch(t.Key)))
            .ToDictionary(
                t => t.Key.StartsWith(LanguageModelPrefix, StringComparison.Ordinal) ? "model." + t.Key[LanguageModelPrefix.Length..] : t.Key,
                t => t.Value,
                StringComparer.Ordinal);
        var weights = new Dictionary<string, WeightInfo>(_text.MapWeights(config, text, options), StringComparer.Ordinal);
        foreach (var (name, tensor) in tensors)
        {
            if (ExpertCount(config) is not null && FusedExperts().Match(name) is { Success: true } m)
            {
                string layer = $"layers.{m.Groups[1].Value}.mlp.";
                var w = m.Groups[2].Value switch
                {
                    "gate.weight" => HfTransforms.Plain(tensor.ToWeightInfo(), layer + "router.weight"),
                    "experts.down_proj" => FusedExpert(config, tensor.ToWeightInfo(), layer + "experts.down.weight"),
                    _ => FusedExpert(config, tensor.ToWeightInfo(), layer + "experts.gate_up.weight"),
                };
                weights[w.Name] = w;
            }
        }

        if (QwenVisionHfMapping.VisionConfig(config) is not null)
        {
            foreach (var (name, tensor) in tensors)
            {
                if (QwenVisionHfMapping.IsVisionTensor(name) && QwenVisionHfMapping.MapVisionTensor(config, name, tensor) is { } w)
                {
                    weights[w.Name] = w;
                }
            }
        }

        return weights;
    }

    /// <summary>A fused expert tensor in the canonical [E, out, in] order (out = 2I for gate_up, H for down).</summary>
    private static WeightInfo FusedExpert(HfConfig config, WeightInfo w, string name)
    {
        long outRows = name.EndsWith("gate_up.weight", StringComparison.Ordinal) ? 2L * config.Int("moe_intermediate_size") : config.Int("hidden_size");
        bool canonical = w.Shape[1].Fixed == outRows && w.Shape[2].Fixed != outRows;
        return canonical ? HfTransforms.Plain(w, name) : HfExpertLayouts.Transposed(w, name);
    }
}
