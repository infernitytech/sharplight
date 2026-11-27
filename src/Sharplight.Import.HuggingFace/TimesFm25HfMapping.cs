using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// TimesFM 2.5 → canonical <c>timesfm2_5</c> (documented on <c>TimesFmAdapter</c>), from either layout:
/// <list type="bullet">
/// <item><description>the timesfm package's (google/timesfm-2.5-200m-pytorch: a config that says model_type <c>timesfm</c> but has
/// <c>quantile_horizon_length</c>; tensors <c>tokenizer.*</c>, <c>stacked_xf.{i}.*</c> with a fused <c>qkv_proj</c>), and</description></item>
/// <item><description>transformers' (<c>TimesFm2_5ModelForPrediction</c>, model_type <c>timesfm2_5</c>, google/timesfm-2.5-200m-transformers; the <c>model.</c>
/// prefix is dropped).</description></item>
/// </list>
/// The forecasting flags default to the model card's recommended <c>ForecastConfig</c> (input normalization, continuous quantile head, flip invariance,
/// non-negativity and the quantile-crossing fix); the context is the context limit less the continuous quantile head's steps (16384 − 1024).
/// </summary>
internal sealed class TimesFm25HfMapping : IHfArchitectureMapping
{
    /// <summary>The model type both layouts import as.</summary>
    public const string ModelType = "timesfm2_5";

    private static readonly Dictionary<string, string> s_packageLayer = new(StringComparer.Ordinal)
    {
        ["attn.qkv_proj.weight"] = "self_attn.qkv_proj.weight",
        ["attn.out.weight"] = "self_attn.o_proj.weight",
        ["attn.query_ln.scale"] = "self_attn.q_norm.weight",
        ["attn.key_ln.scale"] = "self_attn.k_norm.weight",
        ["attn.per_dim_scale.per_dim_scale"] = "self_attn.scaling",
        ["ff0.weight"] = "mlp.ff0.weight",
        ["ff1.weight"] = "mlp.ff1.weight",
        ["pre_attn_ln.scale"] = "input_layernorm.weight",
        ["post_attn_ln.scale"] = "post_attention_layernorm.weight",
        ["pre_ff_ln.scale"] = "pre_feedforward_layernorm.weight",
        ["post_ff_ln.scale"] = "post_feedforward_layernorm.weight",
    };

    /// <inheritdoc/>
    public string Architecture => ModelType;

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = [ModelType];

    /// <inheritdoc/>
    public bool NeedsTokenizer => false;

    /// <summary>Whether <paramref name="config"/> is the timesfm package's TimesFM 2.5 config (model_type <c>timesfm</c> with a <c>quantile_horizon_length</c>).</summary>
    public static bool IsPackageConfig(HfConfig config)
        => config.Root.ValueKind == System.Text.Json.JsonValueKind.Object && config.Root.TryGetProperty("quantile_horizon_length", out _)
           && config.Root.TryGetProperty("model_type", out var type) && type.ValueKind == System.Text.Json.JsonValueKind.String && type.GetString() == "timesfm";

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        string a = Architecture + ".";
        int hidden = c.GetInt("hidden_size") ?? 1280;
        int heads = c.GetInt("num_attention_heads") ?? 16;
        int limit = c.GetInt("context_length", "max_position_embeddings") ?? 16384;
        bool continuous = c.GetBool("use_continuous_quantile_head") ?? true;
        int quantileHorizon = continuous ? c.GetInt("output_quantile_len", "quantile_horizon_length") ?? 1024 : 0;
        return new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "hidden_size"] = hidden,
            [a + "intermediate_size"] = c.GetInt("intermediate_size") ?? hidden,
            [a + "layer_count"] = c.GetInt("num_hidden_layers") ?? 20,
            [a + "attention.head_count"] = heads,
            [a + "attention.head_dim"] = c.GetInt("head_dim") ?? (hidden / heads),
            [a + "norm_eps"] = (float)(c.GetDouble("rms_norm_eps") ?? 1e-6),
            [a + "context_length"] = limit - Math.Max(quantileHorizon, c.GetInt("horizon_length") ?? 128),
            [a + "sequence_limit"] = limit,
            [a + "prediction_length"] = c.GetInt("horizon_length") ?? 128,
            [a + "patch_size"] = c.GetInt("patch_length") ?? 32,
            [a + "quantiles"] = TimesFmHfMapping.Quantiles(c),
            [a + "quantile_horizon"] = quantileHorizon,
            [a + "decode_index"] = c.GetInt("decode_index") ?? 5,
            [a + "rope.base"] = (float)(c.GetDouble("rope_theta") ?? 10000),
            [a + "normalize_inputs"] = c.GetBool("normalize_inputs") ?? true,
            [a + "flip_invariance"] = c.GetBool("force_flip_invariance") ?? true,
            [a + "non_negative"] = c.GetBool("infer_is_positive") ?? true,
            [a + "fix_quantile_crossing"] = c.GetBool("fix_quantile_crossing") ?? true,
        };
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        var collector = new HfWeightCollector();
        foreach (var (name, tensor) in tensors)
        {
            if (Canonical(name) is { } canonical)
            {
                collector.Add(HfTransforms.Plain(tensor.ToWeightInfo(), canonical));
            }
            else
            {
                collector.AddUnmapped(name);
            }
        }

        collector.ThrowIfUnmapped(ModelType);
        return collector.Weights;
    }

    private static string? Canonical(string name)
    {
        if (name.StartsWith("model.", StringComparison.Ordinal))
        {
            return name["model.".Length..];   // transformers: model.input_ff_layer.*, model.layers.{i}.*
        }

        if (name.StartsWith("output_projection_", StringComparison.Ordinal))
        {
            return name.Replace(".hidden_layer.", ".input_layer.", StringComparison.Ordinal);
        }

        if (name.StartsWith("tokenizer.", StringComparison.Ordinal))
        {
            return "input_ff_layer." + name["tokenizer.".Length..].Replace("hidden_layer.", "input_layer.", StringComparison.Ordinal);
        }

        if (name.StartsWith("stacked_xf.", StringComparison.Ordinal))
        {
            int dot = name.IndexOf('.', "stacked_xf.".Length);
            string layer = name["stacked_xf.".Length..dot];
            return s_packageLayer.TryGetValue(name[(dot + 1)..], out var suffix) ? $"layers.{layer}.{suffix}" : null;
        }

        return null;
    }
}
