using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Hugging Face <c>nemotron_h</c> checkpoints → canonical <c>nemotron_h</c> (vocabulary documented on <c>NemotronHAdapter</c>). A = -exp(A_log) and the conv
/// are converted to f32 lazily. <c>nemotron_h_omni</c> (Nemotron 3.5 Super-VL / Bolt, remote code) wraps the same language model below
/// <c>language_model.</c> with its <c>llm_config</c>; its RADIO vision tower, projector (<c>mlp1</c>) and sound encoder are not imported.
/// </summary>
internal sealed partial class NemotronHHfMapping : IHfArchitectureMapping
{
    /// <inheritdoc/>
    public string Architecture => "nemotron_h";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["nemotron_h", "nemotron_h_puzzle", OmniModelType];

    private const string OmniModelType = "nemotron_h_omni";
    private const string OmniLanguageModelPrefix = "language_model.";

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

        var layerTypes = c.GetStringArray("layers_block_type") ?? c.GetStringArray("layer_types") ?? PuzzleBlockTypes(c)
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

    /// <summary>The <c>block_type</c> of every entry of a Puzzle config's <c>block_configs</c>, or null.</summary>
    private static string[]? PuzzleBlockTypes(HfConfig c)
        => c.Get("block_configs") is { ValueKind: JsonValueKind.Array } blocks
            ? blocks.EnumerateArray().Select(block => block.GetProperty("block_type").GetString() ?? "").ToArray()
            : null;

    /// <summary>Per layer, the <c>block_configs</c> entry of a Puzzle config's MoE layers (null for the other layers); null without block_configs.</summary>
    private static JsonElement?[]? PuzzleMoeBlocks(HfConfig c)
        => c.Get("block_configs") is { ValueKind: JsonValueKind.Array } blocks
            ? blocks.EnumerateArray().Select(block => block.GetProperty("block_type").GetString() == "moe" ? block : (JsonElement?)null).ToArray()
            : null;

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
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
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
        int groupCount = c.GetInt("n_group") ?? 1;
        if (groupCount > 1 && (c.GetInt("topk_group") ?? 1) != groupCount)
        {
            throw new NotSupportedException("nemotron_h: group-limited expert routing is not supported yet.");
        }

        if (PuzzleMoeBlocks(c) is { } blocks)
        {
            // Heterogeneous ("Puzzle", Nemotron 3.5 Bolt): every MoE layer has its own expert count, width and selection count; the adapter reads
            // the first two from the weights.
            var used = blocks.Select(block => block is { } b && b.TryGetProperty("num_experts_per_tok", out var k) ? k.GetInt32() : 0).ToArray();
            md[a + "moe.expert_count"] = blocks.Max(block => block is { } b && b.TryGetProperty("n_routed_experts", out var e) ? e.GetInt32() : 0);
            md[a + "moe.expert_used_count"] = used.Max();
            if (used.Where(u => u > 0).Distinct().Count() > 1)
            {
                md[a + "moe.layer_expert_used_counts"] = used;
            }
        }
        else
        {
            md[a + "moe.expert_count"] = c.Int("n_routed_experts");
            md[a + "moe.expert_used_count"] = c.Int("num_experts_per_tok");
        }
        md[a + "moe.normalize_weights"] = c.GetBool("norm_topk_prob") ?? true;
        if (c.GetDouble("routed_scaling_factor") is { } scalingFactor && scalingFactor != 1.0)
        {
            md[a + "moe.weights_scale"] = (float)scalingFactor;
        }
    }

    [GeneratedRegex(@"^backbone\.layers\.(\d+)\.(.+)$")]
    private static partial Regex LayerName();

    [GeneratedRegex(@"^mtp\.layers\.(\d+)\.(.+)$")]
    private static partial Regex MtpName();

    /// <summary>NextN-only tensors of the multi-token-prediction head, HF suffix to canonical suffix (the GGUF mapping's names).</summary>
    private static readonly Dictionary<string, string> s_mtp = new(StringComparer.Ordinal)
    {
        ["enorm.weight"] = "embedding_norm.weight",
        ["hnorm.weight"] = "hidden_norm.weight",
        ["eh_proj.weight"] = "fc.weight",
        ["final_layernorm.weight"] = "output_norm.weight",
    };

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
        ["mixer.fc1_latent_proj.weight"] = "moe.latent_down.weight",
        ["mixer.fc2_latent_proj.weight"] = "moe.latent_up.weight",
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
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        var collector = new HfWeightCollector();
        var experts = new SortedDictionary<(string Prefix, string Kind), SortedDictionary<int, WeightInfo>>();
        bool omni = c.ModelType == OmniModelType;
        foreach (var (rawName, tensor) in tensors.OrderBy(p => p.Key, StringComparer.Ordinal))
        {
            if (omni && !rawName.StartsWith(OmniLanguageModelPrefix, StringComparison.Ordinal))
            {
                continue;   // vision tower, projector, sound encoder
            }

            string name = omni ? rawName[OmniLanguageModelPrefix.Length..] : rawName;
            var w = tensor.ToWeightInfo();
            if (TryMapGlobalTensor(c, name, w, collector))
            {
                continue;
            }

            var m = LayerName().Match(name);
            bool mapped = m.Success
                ? TryMapLayerTensor($"layers.{m.Groups[1].Value}.", m.Groups[2].Value, w, collector, experts)
                : TryMapMtpTensor(name, w, collector, experts);
            if (!mapped)
            {
                collector.AddUnmapped(rawName);
            }
        }

        foreach (var ((prefix, kind), byIndex) in experts)
        {
            if (byIndex.Keys.Max() != byIndex.Count - 1)
            {
                throw new InvalidDataException($"{prefix}: expert indices of {kind} are not contiguous.");
            }

            collector.Add(WeightInfo.Stack($"{prefix}moe.experts.{kind}.weight", [.. byIndex.Values]));
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

    /// <summary>
    /// One tensor of the multi-token-prediction head (<c>num_nextn_predict_layers</c> 1, as Nemotron 3 Ultra and 3.5 ship it):
    /// <c>mtp.layers.0</c> is the NextN input fusion with an attention block, <c>mtp.layers.1</c> a mixture of experts (its norm the
    /// post-attention norm) with the head norm. Both fold into the one canonical block below <c>mtp.layers.0.</c>, as llama.cpp's
    /// converter folds them into one GGUF block; false when <paramref name="name"/> is no such tensor.
    /// </summary>
    private static bool TryMapMtpTensor(
        string name,
        WeightInfo w,
        HfWeightCollector collector,
        SortedDictionary<(string Prefix, string Kind), SortedDictionary<int, WeightInfo>> experts)
    {
        const string Prefix = "mtp.layers.0.";
        var m = MtpName().Match(name);
        if (!m.Success)
        {
            return false;
        }

        string suffix = m.Groups[2].Value;
        if (s_mtp.TryGetValue(suffix, out var nextn))
        {
            collector.Add(HfTransforms.Plain(w, Prefix + nextn));
            return true;
        }

        if (m.Groups[1].Value != "0" && suffix == "norm.weight")
        {
            collector.Add(HfTransforms.Plain(w, Prefix + "post_attention_norm.weight"));
            return true;
        }

        return TryMapLayerTensor(Prefix, suffix, w, collector, experts);
    }

    /// <summary>One <c>backbone.layers.N.*</c> tensor, canonical names below <paramref name="prefix"/>; false when it has no canonical counterpart.</summary>
    private static bool TryMapLayerTensor(
        string prefix,
        string suffix,
        WeightInfo w,
        HfWeightCollector collector,
        SortedDictionary<(string Prefix, string Kind), SortedDictionary<int, WeightInfo>> experts)
    {
        if (ExpertName().Match(suffix) is { Success: true } expertMatch)
        {
            // Per-expert tensors are stacked into one weight per layer and kind once all are seen.
            var key = (prefix, expertMatch.Groups[2].Value);
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
