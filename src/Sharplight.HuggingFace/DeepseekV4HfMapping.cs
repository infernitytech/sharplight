using System.Buffers.Binary;
using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.HuggingFace;

/// <summary>
/// DeepSeek-V4 (HF <c>deepseek_v4</c>) → canonical <c>latent_decoder</c> with compressed-KV attention (<c>attention.compression.*</c>, vocabulary
/// on <c>LatentDecoderAdapter</c>): a single shared key-value head over a sliding window plus compressed entries (layer types
/// <c>sliding_attention</c>, <c>compressed_sparse_attention</c> with overlapping windows and an indexer, <c>heavily_compressed_attention</c>),
/// grouped low-rank output, hyper-connections with Sinkhorn mixing, sqrt-softplus routing with hash-routed first layers (<c>tid2eid</c>) and
/// clamped SwiGLU experts. Tensor names are the native ones (<c>attn.wq_a</c>, <c>ffn.experts.{e}.w1</c>, ...), with or without the
/// <c>model.</c> prefix; the multi-token-prediction layers are not imported.
/// </summary>
internal sealed partial class DeepseekV4HfMapping : IHfArchitectureMapping
{
    /// <inheritdoc/>
    public string Architecture => "latent_decoder";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["deepseek_v4"];

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HfImportOptions options)
    {
        string a = Architecture + ".";
        int layers = c.Int("num_hidden_layers");
        float eps = (float)(c.GetDouble("rms_norm_eps") ?? 1e-6);
        float limit = (float)(c.GetDouble("swiglu_limit") ?? 0);
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = layers,
            [a + "hidden_size"] = c.Int("hidden_size"),
            [a + "norm_eps"] = eps,
            [a + "attention.head_count"] = c.Int("num_attention_heads"),
            [a + "attention.q_rank"] = c.Int("q_lora_rank"),
            [a + "attention.rope_dim"] = RopeDims(c),
            [a + "attention.compression.ratios"] = Ratios(c, layers),
            [a + "attention.compression.window"] = c.Int("sliding_window"),
            [a + "attention.compression.top_k"] = c.GetInt("index_topk") ?? 0,
            [a + "attention.compression.output_groups"] = c.Int("o_groups"),
            [a + "rope.base"] = (float)(c.LayerTypeRopeDouble("main", "rope_theta") ?? c.GetDouble("rope_theta") ?? 10000.0),
            [a + "rope.base_compressed"] = (float)(c.LayerTypeRopeDouble("compress", "rope_theta") ?? c.GetDouble("compress_rope_theta") ?? 160000.0),
            [a + "rope.style"] = "gptj",
            [a + "moe.expert_count"] = c.Int("n_routed_experts"),
            [a + "moe.expert_used_count"] = c.Int("num_experts_per_tok"),
            [a + "moe.router_activation"] = (c.GetString("scoring_func") ?? "sqrtsoftplus") switch
            {
                "sqrtsoftplus" => "sqrt_softplus",
                "sigmoid" => "sigmoid",
                "softmax" => "softmax",
                var other => throw new InvalidDataException($"deepseek_v4: unknown scoring_func '{other}'."),
            },
            [a + "moe.normalize_weights"] = c.GetBool("norm_topk_prob") ?? true,
            [a + "moe.weights_scale"] = (float)(c.GetDouble("routed_scaling_factor") ?? 1.0),
            [a + "hyper_connections.streams"] = c.Int("hc_mult"),
            [a + "hyper_connections.eps"] = (float)(c.GetDouble("hc_eps") ?? 1e-6),
            [a + "hyper_connections.magnitude"] = 2f,
            [a + "hyper_connections.post_offset"] = 0f,
            [a + "hyper_connections.sinkhorn_iterations"] = c.Int("hc_sinkhorn_iters"),
        };
        HfMappingHelpers.AddContextLength(md, a, c);
        if (limit > 0)
        {
            // Experts and the shared expert: min(gate, limit) before the SiLU, up clamped to ±limit.
            md[a + "moe.swiglu_clamp"] = Enumerable.Repeat(limit, layers).ToArray();
            md[a + "moe.swiglu_clamp_input"] = true;
            md[a + "mlp.swiglu_clamp"] = Enumerable.Repeat(limit, layers).ToArray();
            md[a + "mlp.swiglu_clamp_input"] = true;
        }

        return md;
    }

    private static int RopeDims(HfConfig c) => c.GetInt("qk_rope_head_dim")
        ?? (int)(c.Int("head_dim") * (c.GetDouble("partial_rotary_factor") ?? c.LayerTypeRopeDouble("main", "partial_rotary_factor") ?? 1.0));

    /// <summary>Positions per compressed entry of every layer: from <c>layer_types</c> and <c>compress_rates</c>, or the native <c>compress_ratios</c>.</summary>
    /// <exception cref="InvalidDataException">A layer type is unknown.</exception>
    private static int[] Ratios(HfConfig c, int layers)
    {
        if (c.Get("compress_ratios") is { ValueKind: JsonValueKind.Array } native)
        {
            return [.. native.EnumerateArray().Take(layers).Select(e => e.GetInt32())];
        }

        var rates = c.Get("compress_rates");
        int Rate(string type) => rates is { ValueKind: JsonValueKind.Object } r && r.TryGetProperty(type, out var v) ? v.GetInt32()
            : throw new InvalidDataException($"deepseek_v4: compress_rates has no '{type}'.");
        var types = c.GetStringArray("layer_types") ?? throw new InvalidDataException("deepseek_v4: layer_types is missing.");
        return [.. types.Take(layers).Select(t => t switch
        {
            "sliding_attention" => 0,
            "compressed_sparse_attention" or "heavily_compressed_attention" => Rate(t),
            _ => throw new InvalidDataException($"deepseek_v4: unknown layer type '{t}'."),
        })];
    }

    [GeneratedRegex(@"^(?:model\.)?layers\.(\d+)\.(.+)$")]
    private static partial Regex LayerName();

    [GeneratedRegex(@"^ffn\.experts\.(\d+)\.(w1|w2|w3)\.weight$")]
    private static partial Regex ExpertName();

    /// <summary>Per-layer tensors that are only renamed, native suffix to canonical suffix.</summary>
    private static readonly Dictionary<string, string> s_layer = new(StringComparer.Ordinal)
    {
        ["attn_norm.weight"] = "input_norm.weight",
        ["ffn_norm.weight"] = "post_attention_norm.weight",
        ["attn.wq_a.weight"] = "attention.q_down.weight",
        ["attn.q_norm.weight"] = "attention.q_down_norm.weight",
        ["attn.wq_b.weight"] = "attention.q_up.weight",
        ["attn.wkv.weight"] = "attention.kv.weight",
        ["attn.norm.weight"] = "attention.kv_norm.weight",
        ["attn.kv_norm.weight"] = "attention.kv_norm.weight",
        ["attn.wo_b.weight"] = "attention.o.weight",
        ["attn.compressor.wkv.weight"] = "attention.compressor.kv.weight",
        ["attn.compressor.wgate.weight"] = "attention.compressor.gate.weight",
        ["attn.compressor.norm.weight"] = "attention.compressor.norm.weight",
        ["attn.indexer.wq_b.weight"] = "attention.indexer.q_up.weight",
        ["attn.indexer.compressor.wkv.weight"] = "attention.indexer.compressor.kv.weight",
        ["attn.indexer.compressor.wgate.weight"] = "attention.indexer.compressor.gate.weight",
        ["attn.indexer.compressor.norm.weight"] = "attention.indexer.compressor.norm.weight",
        ["ffn.gate.weight"] = "mlp.router.weight",
        ["ffn.shared_experts.w1.weight"] = "mlp.shared.gate.weight",
        ["ffn.shared_experts.w2.weight"] = "mlp.shared.down.weight",
        ["ffn.shared_experts.w3.weight"] = "mlp.shared.up.weight",
    };

    /// <summary>Per-layer tensors converted to f32, native suffix to canonical suffix.</summary>
    private static readonly Dictionary<string, string> s_layerF32 = new(StringComparer.Ordinal)
    {
        ["attn.attn_sink"] = "attention.sinks",
        ["attn.compressor.ape"] = "attention.compressor.position_bias",
        ["attn.indexer.compressor.ape"] = "attention.indexer.compressor.position_bias",
        ["attn.indexer.weights_proj.weight"] = "attention.indexer.head_weights.weight",
        ["ffn.gate.bias"] = "mlp.router_bias",
        ["hc_attn_fn"] = "attention_hc.fn.weight",
        ["hc_attn_base"] = "attention_hc.base",
        ["hc_attn_scale"] = "attention_hc.scale",
        ["hc_ffn_fn"] = "mlp_hc.fn.weight",
        ["hc_ffn_base"] = "mlp_hc.base",
        ["hc_ffn_scale"] = "mlp_hc.scale",
    };

    private static readonly Dictionary<string, string> s_expertKinds = new(StringComparer.Ordinal) { ["w1"] = "gate", ["w2"] = "down", ["w3"] = "up" };

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HfImportOptions options)
    {
        var collector = new HfWeightCollector();
        var experts = new SortedDictionary<(string Prefix, string Kind), SortedDictionary<int, WeightInfo>>();
        int layers = c.Int("num_hidden_layers");
        int groups = c.Int("o_groups");
        foreach (var (name, tensor) in tensors.OrderBy(p => p.Key, StringComparer.Ordinal))
        {
            var w = tensor.ToWeightInfo();
            if (TryMapGlobal(c, name, w, collector) || name.StartsWith("mtp.", StringComparison.Ordinal) || name.StartsWith("model.mtp.", StringComparison.Ordinal))
            {
                continue;   // the multi-token-prediction layers are not imported
            }

            var m = LayerName().Match(name);
            if (!m.Success)
            {
                collector.AddUnmapped(name);
                continue;
            }

            int layer = int.Parse(m.Groups[1].Value);
            string suffix = m.Groups[2].Value;
            string prefix = $"layers.{layer}.";
            if (layer >= layers)
            {
                continue;   // NextN layers after the trunk
            }

            if (ExpertName().Match(suffix) is { Success: true } expert)
            {
                var key = (prefix, s_expertKinds[expert.Groups[2].Value]);
                if (!experts.TryGetValue(key, out var byIndex))
                {
                    experts[key] = byIndex = [];
                }

                byIndex[int.Parse(expert.Groups[1].Value)] = w;
            }
            else if (s_layer.TryGetValue(suffix, out var renamed))
            {
                collector.Add(HfTransforms.Plain(w, prefix + renamed));
            }
            else if (s_layerF32.TryGetValue(suffix, out var widened))
            {
                collector.Add(HfTransforms.ToF32(w, prefix + widened));
            }
            else if (suffix == "attn.wo_a.weight")
            {
                // [G·R, H·D/G] → [G, R, H·D/G]: one "expert" per output group.
                var shape = new Shape(groups, w.Shape[0].Fixed / groups, w.Shape[1].Fixed);
                collector.Add(w.As(prefix + "attention.o_group.weight", shape));
            }
            else if (suffix == "ffn.gate.tid2eid")
            {
                collector.Add(ToI32(w, prefix + "mlp.router.token_experts"));
            }
            else
            {
                collector.AddUnmapped(name);
            }
        }

        foreach (var ((prefix, kind), byIndex) in experts)
        {
            if (byIndex.Keys.Max() != byIndex.Count - 1)
            {
                throw new InvalidDataException($"{prefix}: expert indices of {kind} are not contiguous.");
            }

            collector.Add(WeightInfo.Stack($"{prefix}mlp.experts.{kind}.weight", [.. byIndex.Values]));
        }

        if (CompressedYarn(c) is { } yarn)
        {
            collector.Add(HfMappingHelpers.F32Weight("rope.freq_factors_compressed", yarn.Factors));
        }

        collector.ThrowIfUnmapped(c.ModelType);
        return collector.Weights;
    }

    private static bool TryMapGlobal(HfConfig c, string name, WeightInfo w, HfWeightCollector collector)
    {
        switch (name)
        {
            case "model.embed_tokens.weight" or "embed.weight":
                collector.Add(HfTransforms.Plain(w, "token_embedding.weight"));
                return true;
            case "model.norm.weight" or "norm.weight":
                collector.Add(HfTransforms.Plain(w, "output_norm.weight"));
                return true;
            case "head.weight" or "lm_head.weight":
                if (!(c.GetBool("tie_word_embeddings") ?? false))
                {
                    collector.Add(HfTransforms.Plain(w, "lm_head.weight"));
                }

                return true;
            case "model.hc_head.hc_fn" or "hc_head_fn":
                collector.Add(HfTransforms.ToF32(w, "output_hc.fn.weight"));
                return true;
            case "model.hc_head.hc_base" or "hc_head_base":
                collector.Add(HfTransforms.ToF32(w, "output_hc.base"));
                return true;
            case "model.hc_head.hc_scale" or "hc_head_scale":
                collector.Add(HfTransforms.ToF32(w, "output_hc.scale"));
                return true;
            default:
                return false;
        }
    }

    /// <summary>
    /// The yarn frequency factors of the compressed layers' rope (<c>rope_parameters.compress</c>, or the legacy top-level <c>rope_scaling</c>);
    /// null without yarn. The cos/sin magnitude stays 1 (<c>attention_factor</c> 1, which DeepSeek-V4 sets).
    /// </summary>
    private static RopeScaling.Result? CompressedYarn(HfConfig c)
    {
        JsonElement? json = c.Get("rope_parameters") is { ValueKind: JsonValueKind.Object } rp && rp.TryGetProperty("compress", out var compress)
            ? compress
            : c.Get("rope_scaling") is { ValueKind: JsonValueKind.Object } legacy ? legacy : null;
        if (json is not { } y || !(y.TryGetProperty("rope_type", out var t) || y.TryGetProperty("type", out t)) || t.GetString() != "yarn")
        {
            return null;
        }

        int originalContext = (int)(HfMappingHelpers.OptionalNumber(y, "original_max_position_embeddings") ?? c.GetInt("max_position_embeddings") ?? 4096);
        float ropeBase = (float)(HfMappingHelpers.OptionalNumber(y, "rope_theta") ?? c.GetDouble("compress_rope_theta") ?? 160000.0);
        return RopeScaling.Yarn(RopeDims(c), ropeBase, (float)y.GetProperty("factor").GetDouble(), originalContext,
            (float)HfMappingHelpers.NumberOr(y, "beta_fast", 32), (float)HfMappingHelpers.NumberOr(y, "beta_slow", 1), 1f);
    }

    // An integer table (i64 or i32) as i32.
    private static WeightInfo ToI32(WeightInfo w, string name)
    {
        if (w.Type == DataType.I32)
        {
            return w.As(name);
        }

        return WeightInfo.Derived(name, DataType.I32, w.Shape, () =>
        {
            var values = HfTransforms.DecodeF32(w);
            var bytes = new byte[values.Length * 4];
            for (int i = 0; i < values.Length; i++)
            {
                BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(i * 4), (int)values[i]);
            }

            return bytes;
        });
    }
}
