using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.HuggingFace;

/// <summary>Hugging Face <c>nemotron_h</c> checkpoints → canonical <c>nemotron_h</c> (vocabulary documented on <c>NemotronHAdapter</c>). A = -exp(A_log) and the conv are converted to f32 lazily.</summary>
internal sealed partial class NemotronHHfMapping : IHfArchitectureMapping
{
    /// <inheritdoc/>
    public string Architecture => "nemotron_h";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["nemotron_h"];

    /// <summary>The canonical block kind of every layer, from <c>hybrid_override_pattern</c> or an explicit layer-type list.</summary>
    private static string[] Kinds(HfConfig c)
    {
        if (c.GetString("hybrid_override_pattern") is { Length: > 0 } pattern)
        {
            return pattern.Select(character => character switch
            {
                'M' => "ssm",
                '*' => "attention",
                '-' => "mlp",
                'E' => "moe",
                _ => throw new NotSupportedException($"nemotron_h: unknown hybrid_override_pattern character '{character}'."),
            }).ToArray();
        }

        var layerTypes = c.GetStringArray("layers_block_type") ?? c.GetStringArray("layer_types")
            ?? throw new InvalidDataException("nemotron_h: no hybrid_override_pattern / layers_block_type in config.json.");
        return layerTypes.Select(type => type switch
        {
            "mamba" or "linear_attention" => "ssm",
            "attention" or "full_attention" => "attention",
            "mlp" => "mlp",
            "moe" => "moe",
            _ => throw new NotSupportedException($"nemotron_h: unknown layer type '{type}'."),
        }).ToArray();
    }

    private static string Activation(HfConfig c)
        => (c.GetString("mlp_hidden_act") ?? "relu2") switch
        {
            "relu2" or "relu_squared" => "relu2",
            "silu" => "silu",
            "gelu" => "gelu",
            "relu" => "relu",
            var other => throw new NotSupportedException($"nemotron_h: activation '{other}'."),
        };

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HfImportOptions options)
    {
        string a = Architecture + ".";
        var kinds = Kinds(c);
        int layerCount = c.GetInt("num_hidden_layers") ?? kinds.Length;   // newer configs derive the depth from the layer pattern
        if (kinds.Length != layerCount)
        {
            throw new InvalidDataException("nemotron_h: layer pattern length differs from num_hidden_layers.");
        }

        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = layerCount,
            [a + "hidden_size"] = c.Int("hidden_size"),
            [a + "norm_eps"] = (float)(c.GetDouble("layer_norm_epsilon", "rms_norm_eps") ?? 1e-5),
            [a + "block_kinds"] = kinds,
            [a + "mlp.activation"] = Activation(c),
            [a + "ssm.conv_kernel"] = c.Int("conv_kernel"),
            [a + "ssm.state_size"] = c.Int("ssm_state_size"),
            [a + "ssm.group_count"] = c.Int("n_groups"),
            [a + "ssm.head_count"] = c.Int("mamba_num_heads"),
            [a + "ssm.head_dim"] = c.Int("mamba_head_dim"),
        };
        if (kinds.Contains("attention"))
        {
            int headCount = c.Int("num_attention_heads");
            md[a + "attention.head_count"] = headCount;
            md[a + "attention.kv_head_count"] = c.GetInt("num_key_value_heads") ?? headCount;
            md[a + "attention.head_dim"] = c.GetInt("head_dim") ?? c.Int("hidden_size") / headCount;
        }

        HfMappingHelpers.AddContextLength(md, a, c);

        if (kinds.Contains("moe"))
        {
            AddMixtureOfExperts(md, a, c);
        }

        return md;
    }

    private static void AddMixtureOfExperts(Dictionary<string, object> md, string a, HfConfig c)
    {
        if (c.Get("moe_latent_size") is { ValueKind: not JsonValueKind.Null })
        {
            throw new NotSupportedException("nemotron_h: latent-space experts (moe_latent_size) are not supported yet.");
        }

        int groupCount = c.GetInt("n_group") ?? 1;
        if (groupCount > 1 && (c.GetInt("topk_group") ?? 1) != groupCount)
        {
            throw new NotSupportedException("nemotron_h: group-limited expert routing is not supported yet.");
        }

        md[a + "moe.expert_count"] = c.Int("n_routed_experts");
        md[a + "moe.expert_used_count"] = c.Int("num_experts_per_tok");
        md[a + "moe.normalize_weights"] = c.GetBool("norm_topk_prob") ?? true;
        if (c.GetDouble("routed_scaling_factor") is { } scalingFactor && scalingFactor != 1.0)
        {
            md[a + "moe.weights_scale"] = (float)scalingFactor;
        }
    }

    [GeneratedRegex(@"^backbone\.layers\.(\d+)\.(.+)$")]
    private static partial Regex LayerName();

    [GeneratedRegex(@"^mixer\.experts\.(\d+)\.(up|down)_proj\.weight$")]
    private static partial Regex ExpertName();

    /// <summary>Per-layer tensors that are only renamed, HF suffix to canonical suffix.</summary>
    private static readonly Dictionary<string, string> s_layer = new(StringComparer.Ordinal)
    {
        ["norm.weight"] = "input_norm.weight",
        ["mixer.in_proj.weight"] = "ssm.in.weight",
        ["mixer.out_proj.weight"] = "ssm.out.weight",
        ["mixer.q_proj.weight"] = "attention.q.weight",
        ["mixer.k_proj.weight"] = "attention.k.weight",
        ["mixer.v_proj.weight"] = "attention.v.weight",
        ["mixer.o_proj.weight"] = "attention.o.weight",
        ["mixer.up_proj.weight"] = "mlp.up.weight",
        ["mixer.down_proj.weight"] = "mlp.down.weight",
        ["mixer.gate.weight"] = "moe.router.weight",
        ["mixer.shared_experts.up_proj.weight"] = "moe.shared.up.weight",
        ["mixer.shared_experts.down_proj.weight"] = "moe.shared.down.weight",
    };

    /// <summary>Per-layer tensors converted to F32 (state-space parameters and the router bias), HF suffix to canonical suffix.</summary>
    private static readonly Dictionary<string, string> s_layerF32 = new(StringComparer.Ordinal)
    {
        ["mixer.D"] = "ssm.d",
        ["mixer.dt_bias"] = "ssm.dt_bias",
        ["mixer.conv1d.bias"] = "ssm.conv1d.bias",
        ["mixer.norm.weight"] = "ssm.norm.weight",
        ["mixer.gate.e_score_correction_bias"] = "moe.router_bias",
    };

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HfImportOptions options)
    {
        var collector = new HfWeightCollector();
        var experts = new SortedDictionary<(int Layer, string Kind), SortedDictionary<int, WeightInfo>>();
        foreach (var (name, tensor) in tensors.OrderBy(p => p.Key, StringComparer.Ordinal))
        {
            var w = tensor.ToWeightInfo();
            if (TryMapGlobalTensor(c, name, w, collector))
            {
                continue;
            }

            var m = LayerName().Match(name);
            if (!m.Success || !TryMapLayerTensor(m.Groups[1].Value, m.Groups[2].Value, w, collector, experts))
            {
                collector.AddUnmapped(name);
            }
        }

        foreach (var ((layer, kind), byIndex) in experts)
        {
            if (byIndex.Keys.Max() != byIndex.Count - 1)
            {
                throw new InvalidDataException($"layer {layer}: expert indices of {kind} are not contiguous.");
            }

            collector.Add(WeightInfo.Stack($"layers.{layer}.moe.experts.{kind}.weight", [.. byIndex.Values]));
        }

        collector.ThrowIfUnmapped("nemotron_h");
        return collector.Weights;
    }

    /// <summary>Embedding, final norm and output head; false when <paramref name="name"/> is none of them.</summary>
    private static bool TryMapGlobalTensor(HfConfig c, string name, WeightInfo w, HfWeightCollector collector)
    {
        switch (name)
        {
            case "backbone.embeddings.weight" or "backbone.embedding.weight":
                collector.Add(HfTransforms.Plain(w, "token_embedding.weight"));
                return true;
            case "backbone.norm_f.weight":
                collector.Add(HfTransforms.Plain(w, "output_norm.weight"));
                return true;
            case "lm_head.weight":
                if (!(c.GetBool("tie_word_embeddings") ?? false))
                {
                    collector.Add(HfTransforms.Plain(w, "lm_head.weight"));
                }

                return true;
            default:
                return false;
        }
    }

    /// <summary>One <c>backbone.layers.N.*</c> tensor; false when it has no canonical counterpart.</summary>
    private static bool TryMapLayerTensor(
        string layer,
        string suffix,
        WeightInfo w,
        HfWeightCollector collector,
        SortedDictionary<(int Layer, string Kind), SortedDictionary<int, WeightInfo>> experts)
    {
        string prefix = $"layers.{layer}.";
        if (ExpertName().Match(suffix) is { Success: true } expertMatch)
        {
            // Per-expert tensors are stacked into one weight per layer and kind once all are seen.
            var key = (int.Parse(layer), expertMatch.Groups[2].Value);
            if (!experts.TryGetValue(key, out var byIndex))
            {
                experts[key] = byIndex = [];
            }

            byIndex[int.Parse(expertMatch.Groups[1].Value)] = w;
            return true;
        }

        if (s_layer.TryGetValue(suffix, out var renamed))
        {
            collector.Add(HfTransforms.Plain(w, prefix + renamed));
            return true;
        }

        if (s_layerF32.TryGetValue(suffix, out var f32Name))
        {
            collector.Add(HfTransforms.ToF32(w, prefix + f32Name));
            return true;
        }

        switch (suffix)
        {
            case "mixer.A_log":
                collector.Add(HfTransforms.ToF32(w, prefix + "ssm.a", map: static v => -MathF.Exp(v)));   // A = -exp(A_log)
                return true;
            case "mixer.conv1d.weight":
                collector.Add(HfTransforms.ToF32(w, prefix + "ssm.conv1d.weight", new Shape(w.Shape[0].Fixed, w.Shape[w.Shape.Rank - 1].Fixed)));
                return true;
            default:
                return false;
        }
    }
}
