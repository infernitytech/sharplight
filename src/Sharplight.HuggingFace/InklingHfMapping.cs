using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>inkling_mm_model</c> (Thinking Machines Inkling, the text decoder of the published checkpoint layout <c>model.llm.*</c>) →
/// canonical <c>hybrid</c> (documented on <c>HybridAdapter</c> and <c>HybridAttention</c>): attention without rotary embeddings, sliding-window
/// layers with their own key/value head count, per-head q/k norms with scale 1/head_dim, residual short convolutions of the keys, values, attention
/// output and MLP output, the query-conditioned relative position bias, the log-floor query temperature on the full layers, dense MLPs in the
/// first <c>dense_mlp_idx</c> layers and sigmoid-routed MoE layers whose shared experts are scored by the router (<c>moe.shared_sink_count</c>);
/// the embeddings are RMS-normalized, the logits divided by <c>logits_mup_width_multiplier</c> and cut to <c>unpadded_vocab_size</c>.
/// </summary>
/// <remarks>
/// The fused gate/up projections (<c>w13</c>) interleave gate and up rows; they become separate gate and up weights (zero-copy row views).
/// The vision and audio towers and the multi-token-prediction layers are not mapped.
/// </remarks>
internal sealed partial class InklingHfMapping : IHfArchitectureMapping
{
    /// <inheritdoc/>
    public string Architecture => "hybrid";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["inkling_mm_model"];

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig c)
    {
        if ((c.GetInt("swa_head_dim") ?? c.Int("head_dim")) != c.Int("head_dim"))
        {
            yield return "inkling: sliding-window layers with another head size than the full layers are not supported.";
        }

        if (c.GetString("gate_activation") is { } gate && gate != "sigmoid")
        {
            yield return $"inkling: gate_activation '{gate}' is not supported (only sigmoid).";
        }

        if (c.GetBool("shared_expert_sink") == false || c.GetBool("norm_after_topk") == false || c.GetBool("use_sconv") == false
            || c.GetBool("use_embed_norm") == false)
        {
            yield return "inkling: only the published variant (shared-expert sinks, normalized top-k, short convolutions, embedding norm) is supported.";
        }

        if (c.GetDouble("final_logit_softcapping") is { } cap && cap != 0)
        {
            yield return "inkling: final_logit_softcapping is not supported.";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HfImportOptions options)
    {
        string a = Architecture + ".";
        int layers = c.Int("num_hidden_layers");
        int heads = c.Int("num_attention_heads");
        int kvHeads = c.GetInt("num_key_value_heads") ?? heads;
        bool[] sliding = SlidingLayers(c, layers);
        int vocab = c.Int("vocab_size");
        int unpadded = c.GetInt("unpadded_vocab_size") ?? vocab;
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = layers,
            [a + "hidden_size"] = c.Int("hidden_size"),
            [a + "norm_eps"] = (float)(c.GetDouble("rms_norm_eps") ?? 1e-6),
            [a + "logit_scale"] = (float)(1.0 / (c.GetDouble("logits_mup_width_multiplier") ?? 1.0)),
            [a + "attention.head_count"] = heads,
            [a + "attention.kv_head_count"] = kvHeads,
            [a + "attention.head_dim"] = c.Int("head_dim"),
            [a + "attention.scale_by_head_dim"] = true,
            [a + "attention.sliding_window"] = c.GetInt("sliding_window_size", "sliding_window") ?? 512,
            [a + "attention.sliding_layers"] = sliding.Select(s => s ? 1 : 0).ToArray(),
            [a + "attention.layer_head_counts"] = sliding.Select(s => s ? c.GetInt("swa_num_attention_heads") ?? heads : heads).ToArray(),
            [a + "attention.layer_kv_head_counts"] = sliding.Select(s => s ? c.GetInt("swa_num_key_value_heads") ?? kvHeads : kvHeads).ToArray(),
            [a + "mlp.activation"] = c.GetString("hidden_act") ?? "silu",
            [a + "moe.expert_count"] = c.Int("n_routed_experts", "num_local_experts"),
            [a + "moe.expert_used_count"] = c.Int("num_experts_per_tok"),
            [a + "moe.router_activation"] = "sigmoid",
            [a + "moe.normalize_weights"] = true,
            [a + "moe.weights_scale"] = (float)(c.GetDouble("route_scale") ?? 1.0),
            [a + "moe.shared_sink_count"] = c.GetInt("n_shared_experts") ?? 0,
        };
        if (c.GetDouble("log_scaling_n_floor") is { } floor && floor > 0)
        {
            md[a + "attention.log_scaling_floor"] = (float)floor;
            md[a + "attention.log_scaling_alpha"] = (float)(c.GetDouble("log_scaling_alpha") ?? 0.1);
        }

        if (unpadded < vocab)
        {
            md[a + "logits.suppressed_ranges"] = new[] { unpadded, vocab };
        }

        HfMappingHelpers.AddContextLength(md, a, c, "model_max_length", "max_position_embeddings");
        return md;
    }

    /// <summary>Per-layer sliding flags: <c>local_layer_ids</c> when given, else <c>layer_types</c>, else every layer but each sixth.</summary>
    private static bool[] SlidingLayers(HfConfig c, int layers)
    {
        if (c.Get("local_layer_ids") is { ValueKind: System.Text.Json.JsonValueKind.Array } ids)
        {
            var local = ids.EnumerateArray().Select(e => e.GetInt32()).ToHashSet();
            return [.. Enumerable.Range(0, layers).Select(local.Contains)];
        }

        if (c.GetStringArray("layer_types") is { } types)
        {
            return [.. types.Select(t => t == "hybrid_sliding")];
        }

        return [.. Enumerable.Range(0, layers).Select(i => (i + 1) % 6 != 0)];
    }

    [GeneratedRegex(@"^model\.llm\.layers\.(\d+)\.(.+)$")]
    private static partial Regex LayerName();

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HfImportOptions options)
    {
        var collector = new HfWeightCollector();
        foreach (var (name, tensor) in tensors.OrderBy(p => p.Key, StringComparer.Ordinal))
        {
            if (name.StartsWith("model.mtp.", StringComparison.Ordinal) || name.StartsWith("model.visual.", StringComparison.Ordinal)
                || name.StartsWith("model.audio.", StringComparison.Ordinal))
            {
                continue;
            }

            var w = tensor.ToWeightInfo();
            var m = LayerName().Match(name);
            bool mapped = m.Success ? TryMapLayerTensor($"layers.{m.Groups[1].Value}.", m.Groups[2].Value, w, collector) : TryMapGlobalTensor(name, w, collector);
            if (!mapped)
            {
                collector.AddUnmapped(name);
            }
        }

        collector.ThrowIfUnmapped("inkling_mm_model");
        return collector.Weights;
    }

    private static bool TryMapGlobalTensor(string name, WeightInfo w, HfWeightCollector collector)
    {
        string? canonical = name switch
        {
            "model.llm.embed.weight" => "token_embedding.weight",
            "model.llm.embed_norm.weight" => "embedding_norm.weight",
            "model.llm.norm.weight" => "output_norm.weight",
            "model.llm.unembed.weight" => "lm_head.weight",
            _ => null,
        };
        if (canonical is null)
        {
            return false;
        }

        collector.Add(HfTransforms.Plain(w, canonical));
        return true;
    }

    /// <summary>Per-layer tensors that are only renamed (1-D ones widened to F32), checkpoint suffix to canonical suffix.</summary>
    private static readonly Dictionary<string, string> s_renamed = new(StringComparer.Ordinal)
    {
        ["attn_norm.weight"] = "input_norm.weight",
        ["mlp_norm.weight"] = "post_attention_norm.weight",
        ["attn.wq_du.weight"] = "attention.q.weight",
        ["attn.wk_dv.weight"] = "attention.k.weight",
        ["attn.wv_dv.weight"] = "attention.v.weight",
        ["attn.wr_du.weight"] = "attention.relative.weight",
        ["attn.wo_ud.weight"] = "attention.o.weight",
        ["attn.q_norm.weight"] = "attention.q_norm.weight",
        ["attn.k_norm.weight"] = "attention.k_norm.weight",
        ["mlp.w2_md.weight"] = "mlp.down.weight",
        ["mlp.global_scale"] = "mlp.output_scale",
        ["mlp.gate.global_scale"] = "mlp.output_scale",
        ["mlp.gate.weight"] = "mlp.router.weight",
        ["mlp.gate.bias"] = "mlp.router_bias",
        ["mlp.experts.w2_weight"] = "mlp.experts.down.weight",
        ["mlp.shared_experts.shared_w2_weight"] = "mlp.shared_sinks.down.weight",
    };

    /// <summary>Depthwise short-convolution kernels [C, 1, K], checkpoint suffix to canonical suffix (stored F32 [C, K]).</summary>
    private static readonly Dictionary<string, string> s_convolutions = new(StringComparer.Ordinal)
    {
        ["attn.k_sconv.weight"] = "attention.k_conv.weight",
        ["attn.v_sconv.weight"] = "attention.v_conv.weight",
        ["attn_sconv.weight"] = "post_mixer_conv.weight",
        ["mlp_sconv.weight"] = "mlp.out_conv.weight",
    };

    /// <summary>Fused interleaved gate/up projections, checkpoint suffix to the canonical prefix of the separate gate and up weights.</summary>
    private static readonly Dictionary<string, string> s_fusedGateUp = new(StringComparer.Ordinal)
    {
        ["mlp.w13_dn.weight"] = "mlp.",
        ["mlp.experts.w13_weight"] = "mlp.experts.",
        ["mlp.shared_experts.shared_w13_weight"] = "mlp.shared_sinks.",
    };

    private static bool TryMapLayerTensor(string prefix, string suffix, WeightInfo w, HfWeightCollector collector)
    {
        if (s_renamed.TryGetValue(suffix, out var renamed))
        {
            collector.Add(HfTransforms.Plain(w, prefix + renamed));
            return true;
        }

        if (s_convolutions.TryGetValue(suffix, out var convolution))
        {
            collector.Add(HfTransforms.ToF32(w, prefix + convolution, new Shape(w.Shape[0].Fixed, w.Shape[w.Shape.Rank - 1].Fixed)));
            return true;
        }

        if (s_fusedGateUp.TryGetValue(suffix, out var fused))
        {
            collector.Add(Deinterleaved(w, prefix + fused + "gate.weight", 0));
            collector.Add(Deinterleaved(w, prefix + fused + "up.weight", 1));
            return true;
        }

        if (suffix == "attn.rel_logits_proj.proj")
        {
            // [R, E] → [E, R]: one row of R mixing weights per backward distance.
            long r = w.Shape[0].Fixed;
            long extent = w.Shape[1].Fixed;
            var transposed = HfTransforms.ToF32(w, prefix + "attention.relative_bias.weight", new Shape(1, r * extent))
                .TransposeRowBlocks(prefix + "attention.relative_bias.weight", r, extent);
            collector.Add(transposed.As(prefix + "attention.relative_bias.weight", new Shape(extent, r)));
            return true;
        }

        return false;
    }

    /// <summary>
    /// The gate (<paramref name="parity"/> 0, even rows) or up (1, odd rows) half of a fused interleaved projection [2N, H] or, per expert,
    /// [E, 2N, H]: a zero-copy gather of its rows.
    /// </summary>
    private static WeightInfo Deinterleaved(WeightInfo w, string name, int parity)
    {
        long experts = w.Shape.Rank == 3 ? w.Shape[0].Fixed : 1;
        long pairs = w.Shape[w.Shape.Rank - 2].Fixed / 2;
        long columns = w.Shape[w.Shape.Rank - 1].Fixed;
        var rows = w.As(w.Name, new Shape(experts * pairs * 2, columns));
        var blocks = new List<(long FirstRow, long Rows)>();
        for (long e = 0; e < experts; e++)
        {
            for (long i = 0; i < pairs; i++)
            {
                blocks.Add((e * pairs * 2 + 2 * i + parity, 1));
            }
        }

        var gathered = rows.GatherRows(name, blocks);
        return w.Shape.Rank == 3 ? gathered.As(name, new Shape(experts, pairs, columns)) : gathered;
    }
}
