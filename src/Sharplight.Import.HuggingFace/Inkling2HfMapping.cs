using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Hugging Face <c>inkling2_loop</c> (Thinking Machines Inkling 2 Mini; transformers 5.24 <c>Inkling2LoopForConditionalGeneration</c>, the text
/// decoder of the published checkpoint layout <c>model.llm.*</c>) → canonical <c>block_residual_decoder</c> (documented on
/// <c>BlockResidualDecoderAdapter</c>): block attention residuals over normalized rows, rope-less attention with per-token sinks, an output gate,
/// L2-clipped queries, keys and relative states, value RMS norms, the query-conditioned relative position bias, the log-floor scaling on the
/// full layers, static short convolutions of the sliding layers' keys and values and per-token ones of the full layers', residual convolutions
/// of the attention and MLP outputs, sigmoid-routed experts with a soft-capped SwiGLU, a core of layers run several times that exchange
/// messages between passes, the embedding norm, a final logit soft cap and the unpadded vocabulary.
/// </summary>
/// <remarks>
/// The expert gate/up projections (<c>w13</c>) interleave gate and up rows; they become separate gate and up weights (zero-copy row views). The
/// release's quantized experts (NVFP4 with a per-expert <c>amax</c> / <c>outer_scale_denominator</c> factor, MXFP8) arrive already converted
/// by the quantized-weight readers, renamed from <c>w13_weight</c> to <c>w13.weight</c>. The static taps of the per-token convolutions
/// (<c>static_w</c>, [K, C] indexed by backward distance) become kernels [C, K] in convolution order (oldest tap first), and the rows of their
/// tap projections (<c>weight_proj</c>, [heads·K, hidden], distance-major per head) are reordered the same way. The two message-query tensors
/// of the loop ([2·loop layers, hidden] each: query and MLP queries, key and value queries) are kept as [loop layers, 2, hidden] views. The
/// vision tower and the multi-token-prediction layers (<c>mtp.safetensors</c>) are not mapped.
/// </remarks>
internal sealed partial class Inkling2HfMapping : IHfArchitectureMapping
{
    /// <inheritdoc/>
    public string Architecture => "block_residual_decoder";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["inkling2_loop", "inkling2_loop_text"];

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig c)
    {
        if ((c.GetInt("swa_head_dim") ?? c.Int("head_dim")) != c.Int("head_dim"))
        {
            yield return "inkling2: sliding-window layers with another head size than the full layers are not supported.";
        }

        if (c.GetString("scoring_func") is { } scoring && scoring != "sigmoid")
        {
            yield return $"inkling2: scoring_func '{scoring}' is not supported (only sigmoid).";
        }

        if ((c.GetInt("n_shared_experts") ?? 0) != 0)
        {
            yield return "inkling2: shared experts are not supported.";
        }

        if (c.GetBool("attn_res_normalize_values") == false || c.GetBool("use_value_norm_no_gamma") == false)
        {
            yield return "inkling2: only the published variant (normalized attention-residual values, weight-less value norms) is supported.";
        }

        if (c.Get("loop_config") is { ValueKind: JsonValueKind.Object } loop && loop.TryGetProperty("is_varloop", out var varloop)
            && varloop.ValueKind == JsonValueKind.True)
        {
            yield return "inkling2: variable loop counts (is_varloop) are not supported.";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        string a = Architecture + ".";
        int layers = c.Int("num_hidden_layers");
        int hidden = c.Int("hidden_size");
        int heads = c.Int("num_attention_heads");
        int kvHeads = c.GetInt("num_key_value_heads") ?? heads;
        bool[] sliding = SlidingLayers(c, layers);
        int vocab = c.Int("vocab_size");
        int unpadded = c.GetInt("unpadded_vocab_size") ?? vocab;
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = layers,
            [a + "hidden_size"] = hidden,
            [a + "norm_eps"] = (float)(c.GetDouble("rms_norm_eps") ?? 1e-6),
            [a + "attention.head_count"] = heads,
            [a + "attention.kv_head_count"] = kvHeads,
            [a + "attention.head_dim"] = c.Int("head_dim"),
            [a + "attention.sliding_window"] = c.GetInt("sliding_window_size", "sliding_window") ?? 1024,
            [a + "attention.sliding_layers"] = sliding.Select(s => s ? 1 : 0).ToArray(),
            [a + "attention.layer_head_counts"] = sliding.Select(s => s ? c.GetInt("swa_num_attention_heads") ?? heads : heads).ToArray(),
            [a + "attention.layer_kv_head_counts"] = sliding.Select(s => s ? c.GetInt("swa_num_key_value_heads") ?? kvHeads : kvHeads).ToArray(),
            [a + "attention.qk_clip"] = (float)(c.GetDouble("qk_l2_clip_threshold") ?? 22.627416997969522),
            [a + "attention.relative_clip"] = (float)(c.GetDouble("rel_l2_clip_threshold") ?? 8.0),
            [a + "attention_residuals.block_size"] = c.GetInt("n_layers_per_attn_res_block") ?? 6,
            [a + "attention_residuals.normalized_values"] = true,
            [a + "attention_residuals.score_scale"] = (float)(1.0 / Math.Sqrt(hidden)),
            [a + "mlp.activation"] = "situ",
            [a + "mlp.situ_beta"] = (float)(c.GetDouble("mlp_gate_softcap") ?? 4.0),
            [a + "mlp.situ_linear_beta"] = (float)(c.GetDouble("mlp_up_softcap") ?? 25.0),
            [a + "moe.expert_count"] = c.Int("n_routed_experts", "num_local_experts"),
            [a + "moe.expert_used_count"] = c.Int("num_experts_per_tok"),
            [a + "moe.router_activation"] = "sigmoid",
            [a + "moe.normalize_weights"] = true,
            [a + "moe.weights_scale"] = (float)(c.GetDouble("route_scale", "routed_scaling_factor") ?? 1.0),
            [a + "logits.softcap"] = (float)(c.GetDouble("final_logit_softcapping") ?? 0.0),
        };
        if (c.GetDouble("log_scaling_n_floor") is { } floor && floor > 0)
        {
            md[a + "attention.log_scaling_floor"] = (float)floor;
            md[a + "attention.log_scaling_alpha"] = (float)(c.GetDouble("log_scaling_alpha") ?? 0.1);
        }

        if (c.Get("loop_config") is { ValueKind: JsonValueKind.Object } loop)
        {
            md[a + "loop.first_layer"] = loop.TryGetProperty("core_start", out var start) ? start.GetInt32() : 6;
            md[a + "loop.end_layer"] = loop.TryGetProperty("core_end", out var end) ? end.GetInt32() : 18;
            md[a + "loop.passes"] = loop.TryGetProperty("num_passes", out var passes) ? passes.GetInt32() : 3;
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
        if (c.Get("local_layer_ids") is { ValueKind: JsonValueKind.Array } ids)
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

    [GeneratedRegex(@"^model\.llm\.attn_res\.(attn|mlp)_queries\.(\d+)$")]
    private static partial Regex ResidualQueryName();

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        var collector = new HfWeightCollector();
        int hidden = c.Int("hidden_size");
        int kernel = c.GetInt("sconv_kernel_size", "conv_kernel_size") ?? 4;
        foreach (var (name, tensor) in tensors.OrderBy(p => p.Key, StringComparer.Ordinal))
        {
            if (name.StartsWith("model.mtp.", StringComparison.Ordinal) || name.StartsWith("model.visual.", StringComparison.Ordinal)
                || name.StartsWith("mtp.", StringComparison.Ordinal))
            {
                continue;
            }

            var w = tensor.ToWeightInfo();
            var m = LayerName().Match(name);
            bool mapped = m.Success ? TryMapLayerTensor($"layers.{m.Groups[1].Value}.", m.Groups[2].Value, w, kernel, collector)
                : TryMapGlobalTensor(name, w, hidden, collector);
            if (!mapped)
            {
                collector.AddUnmapped(name);
            }
        }

        collector.ThrowIfUnmapped("inkling2_loop");
        return collector.Weights;
    }

    private static bool TryMapGlobalTensor(string name, WeightInfo w, int hidden, HfWeightCollector collector)
    {
        if (ResidualQueryName().Match(name) is { Success: true } query)
        {
            // attn_queries.i belongs to layer i + 1 (layer 0 reads its input unmixed); [H] or [3, H] (one query each for q, k and v).
            int index = int.Parse(query.Groups[2].Value, System.Globalization.CultureInfo.InvariantCulture);
            bool attention = query.Groups[1].Value == "attn";
            long rows = w.Shape.Rank == 2 ? w.Shape[0].Fixed : 1;
            string canonical = attention ? $"layers.{index + 1}.attention_res.score.weight" : $"layers.{index}.mlp_res.score.weight";
            collector.Add(HfTransforms.ToF32(w, canonical, new Shape(rows, hidden)));
            return true;
        }

        switch (name)
        {
            case "model.llm.attn_res.output_query":
                collector.Add(HfTransforms.ToF32(w, "output_res.score.weight", new Shape(hidden)));
                return true;
            case "model.llm.middle_cycle_cross_pass_queries":
            case "model.llm.middle_cycle_cross_pass_kv_queries":
            {
                // [2·L, H]: rows 2j and 2j + 1 are loop layer j's query and MLP (or key and value) message queries.
                string canonical = name.EndsWith("kv_queries", StringComparison.Ordinal) ? "loop.messages.key_value.weight" : "loop.messages.query_mlp.weight";
                collector.Add(HfTransforms.ToF32(w, canonical, new Shape(w.Shape[0].Fixed / 2, 2, hidden)));
                return true;
            }
        }

        string? renamed = name switch
        {
            "model.llm.embed.weight" => "token_embedding.weight",
            "model.llm.embed_norm.weight" => "embedding_norm.weight",
            "model.llm.norm.weight" => "output_norm.weight",
            "model.llm.unembed.weight" => "lm_head.weight",
            _ => null,
        };
        if (renamed is null)
        {
            return false;
        }

        collector.Add(HfTransforms.Plain(w, renamed));
        return true;
    }

    /// <summary>Per-layer tensors that are only renamed (1-D ones widened to F32), checkpoint suffix to canonical suffix.</summary>
    private static readonly Dictionary<string, string> s_renamed = new(StringComparer.Ordinal)
    {
        ["attn_norm.weight"] = "input_norm.weight",
        ["q_input_norm.weight"] = "q_input_norm.weight",
        ["k_input_norm.weight"] = "k_input_norm.weight",
        ["v_input_norm.weight"] = "v_input_norm.weight",
        ["mlp_norm.weight"] = "post_attention_norm.weight",
        ["attn.wq_du.weight"] = "attention.q.weight",
        ["attn.wk_dv.weight"] = "attention.k.weight",
        ["attn.wv_dv.weight"] = "attention.v.weight",
        ["attn.wr_du.weight"] = "attention.relative.weight",
        ["attn.wo_ud.weight"] = "attention.o.weight",
        ["attn.gate_attn.weight"] = "attention.gate.weight",
        ["attn.attn_sink_proj.weight"] = "attention.sinks.weight",
        ["mlp.gate.weight"] = "mlp.router.weight",
        ["mlp.gate.bias"] = "mlp.router_bias",
        ["mlp.experts.w2_weight"] = "mlp.experts.down.weight",
        ["mlp.experts.w2.weight"] = "mlp.experts.down.weight",
    };

    /// <summary>Per-head scales of the L2 clips ([heads, dims], stored F32).</summary>
    private static readonly Dictionary<string, string> s_clipScales = new(StringComparer.Ordinal)
    {
        ["attn.q_l2_clip.scale"] = "attention.q_clip.scale",
        ["attn.rel_l2_clip.scale"] = "attention.relative_clip.scale",
    };

    /// <summary>Depthwise short-convolution kernels [C, 1, K], checkpoint suffix to canonical suffix (stored F32 [C, K]).</summary>
    private static readonly Dictionary<string, string> s_convolutions = new(StringComparer.Ordinal)
    {
        ["attn.k_sconv.weight"] = "attention.k_conv.weight",
        ["attn.v_sconv.weight"] = "attention.v_conv.weight",
        ["attn_sconv.weight"] = "post_mixer_conv.weight",
        ["mlp_sconv.weight"] = "mlp.out_conv.weight",
    };

    /// <summary>The message gates of the loop layers: checkpoint controller name to canonical gate name.</summary>
    private static readonly Dictionary<string, string> s_controllers = new(StringComparer.Ordinal)
    {
        ["attn"] = "q", ["k"] = "k", ["v"] = "v", ["mlp"] = "mlp",
    };

    [GeneratedRegex(@"^controllers\.(attn|k|v|mlp)\.(down|forget|gain)_weight$")]
    private static partial Regex ControllerName();

    private static bool TryMapLayerTensor(string prefix, string suffix, WeightInfo w, int kernel, HfWeightCollector collector)
    {
        if (s_renamed.TryGetValue(suffix, out var renamed))
        {
            collector.Add(HfTransforms.Plain(w, prefix + renamed));
            return true;
        }

        if (s_clipScales.TryGetValue(suffix, out var clip))
        {
            collector.Add(HfTransforms.ToF32(w, prefix + clip));
            return true;
        }

        if (s_convolutions.TryGetValue(suffix, out var convolution))
        {
            collector.Add(HfTransforms.ToF32(w, prefix + convolution, new Shape(w.Shape[0].Fixed, w.Shape[w.Shape.Rank - 1].Fixed)));
            return true;
        }

        if (ControllerName().Match(suffix) is { Success: true } controller)
        {
            collector.Add(w.As($"{prefix}message_gate.{s_controllers[controller.Groups[1].Value]}.{controller.Groups[2].Value}.weight"));
            return true;
        }

        switch (suffix)
        {
            case "attn.k_sconv.static_w":
            case "attn.v_sconv.static_w":
                collector.Add(StaticTaps(w, prefix + (suffix[5] == 'k' ? "attention.k_conv.weight" : "attention.v_conv.weight")));
                return true;
            case "attn.k_sconv.weight_proj":
            case "attn.v_sconv.weight_proj":
                collector.Add(TapProjection(w, prefix + (suffix[5] == 'k' ? "attention.k_conv.taps.weight" : "attention.v_conv.taps.weight"), kernel));
                return true;
            case "mlp.experts.w13_weight":
            case "mlp.experts.w13.weight":
                collector.Add(Deinterleaved(w, prefix + "mlp.experts.gate.weight", 0));
                collector.Add(Deinterleaved(w, prefix + "mlp.experts.up.weight", 1));
                return true;
            case "attn.rel_logits_proj.proj":
            {
                // [R, E] → [E, R]: one row of R mixing weights per backward distance.
                long r = w.Shape[0].Fixed;
                long extent = w.Shape[1].Fixed;
                string name = prefix + "attention.relative_bias.weight";
                var transposed = HfTransforms.ToF32(w, name, new Shape(1, r * extent)).TransposeRowBlocks(name, r, extent);
                collector.Add(transposed.As(name, new Shape(extent, r)));
                return true;
            }
        }

        return false;
    }

    /// <summary>The static taps [K, C] of a per-token convolution, indexed by backward distance, as a kernel [C, K] in convolution order (F32).</summary>
    private static WeightInfo StaticTaps(WeightInfo w, string name)
    {
        long taps = w.Shape[0].Fixed;
        long channels = w.Shape[1].Fixed;
        return WeightInfo.Derived(name, DataType.F32, new Shape(channels, taps), () =>
        {
            var values = HfTransforms.DecodeF32(w);
            var kernel = new float[values.Length];
            for (long lag = 0; lag < taps; lag++)
            {
                for (long c = 0; c < channels; c++)
                {
                    kernel[c * taps + (taps - 1 - lag)] = values[lag * channels + c];
                }
            }

            var bytes = new byte[kernel.Length * 4];
            Buffer.BlockCopy(kernel, 0, bytes, 0, bytes.Length);
            return bytes;
        });
    }

    /// <summary>The tap projection [heads·K, hidden] (rows distance-major per head) with each head's rows in convolution order (a row gather).</summary>
    private static WeightInfo TapProjection(WeightInfo w, string name, int kernel)
    {
        long heads = w.Shape[0].Fixed / kernel;
        var blocks = new List<(long FirstRow, long Rows)>();
        for (long g = 0; g < heads; g++)
        {
            for (long j = 0; j < kernel; j++)
            {
                blocks.Add((g * kernel + (kernel - 1 - j), 1));
            }
        }

        return w.GatherRows(name, blocks);
    }

    /// <summary>
    /// The gate (<paramref name="parity"/> 0, even rows) or up (1, odd rows) half of a fused interleaved projection [E, 2N, H]: a zero-copy
    /// gather of its rows.
    /// </summary>
    private static WeightInfo Deinterleaved(WeightInfo w, string name, int parity)
    {
        long experts = w.Shape[0].Fixed;
        long pairs = w.Shape[1].Fixed / 2;
        long columns = w.Shape[2].Fixed;
        var rows = w.As(w.Name, new Shape(experts * pairs * 2, columns));
        var blocks = new List<(long FirstRow, long Rows)>();
        for (long e = 0; e < experts; e++)
        {
            for (long i = 0; i < pairs; i++)
            {
                blocks.Add((e * pairs * 2 + 2 * i + parity, 1));
            }
        }

        return rows.GatherRows(name, blocks).As(name, new Shape(experts, pairs, columns));
    }
}
