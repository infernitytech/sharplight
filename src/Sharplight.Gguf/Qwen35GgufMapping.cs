using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>qwen35</c> (Qwen3.5 / 3.6 / 3.8 dense hybrid), <c>qwen35moe</c> (the MoE variants) and <c>qwen3next</c>
/// (Qwen3-Next / Qwen3-Coder-Next: plain partial RoPE, fused <c>ssm_ba</c>, value heads grouped by key head) and
/// <c>qwen4exp</c> (Qwen3.8-Flash-Next: hyper-connections, hashed n-gram embedding, QSA indexer, sigmoid linear-attention
/// output gate) and <c>clover_moe</c> (Clover: the qwen35moe layers with plain 1-D rope and the linear-attention layers given by
/// <c>attention.recurrent_layers</c> or <c>full_attention_interval</c>) → canonical <c>qwen3_5</c>.
/// </summary>
internal sealed class Qwen35GgufMapping : IGgufArchitectureMapping, IGgufSplitMapping
{
    private const int RopeSectionCount = 4;
    private const string Qwen3NextArchitecture = "qwen3next";
    private const string Qwen4ExpArchitecture = "qwen4exp";
    private const string SharedExpertGate = "mlp.shared.input_gate.weight";

    /// <inheritdoc/>
    public IReadOnlyList<string> GgufArchitectures { get; } = ["qwen35", "qwen35moe", Qwen3NextArchitecture, Qwen4ExpArchitecture, "clover_moe"];

    /// <inheritdoc/>
    public string Architecture => "qwen3_5";

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(GgufMetadataView g)
    {
        int nextnLayers = g.Int("nextn_predict_layers", 0);
        int valueHeads = g.Int("ssm.time_step_rank");
        string a = Architecture + ".";
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = g.Int("block_count") - nextnLayers,
            [a + "hidden_size"] = g.Int("embedding_length"),
            [a + "ffn_hidden_size"] = g.Int("feed_forward_length", g.Int("expert_feed_forward_length", 0)),
            [a + "norm_eps"] = g.Float("attention.layer_norm_rms_epsilon"),
            [a + "attention.head_count"] = g.Int("attention.head_count"),
            [a + "attention.kv_head_count"] = g.Int("attention.head_count_kv"),
            [a + "attention.head_dim"] = g.Int("attention.key_length"),
            [a + "rope.dim"] = g.Int("rope.dimension_count"),
            [a + "full_attention_interval"] = g.Int("full_attention_interval", 4),
            [a + "linear_attention.conv_kernel"] = g.Int("ssm.conv_kernel"),
            [a + "linear_attention.key_head_dim"] = g.Int("ssm.state_size"),
            [a + "linear_attention.key_head_count"] = g.Int("ssm.group_count"),
            [a + "linear_attention.value_head_count"] = valueHeads,
            [a + "linear_attention.value_head_dim"] = g.Int("ssm.inner_size") / valueHeads,
            [a + "mtp.layer_count"] = nextnLayers,
        };
        GgufMappingHelpers.AddContextLength(md, a, g);
        CheckRecurrentLayers(g, (int)md[a + "layer_count"], (int)md[a + "full_attention_interval"]);
        if (g.Has("expert_count"))
        {
            md[a + "moe.expert_count"] = g.Int("expert_count");
            md[a + "moe.expert_used_count"] = g.Int("expert_used_count");
            md[a + "moe.normalize_weights"] = true;
        }

        if (IsQwen3Next(g))
        {
            md[a + "linear_attention.grouped_value_heads"] = true;
        }

        if (g.GgufArchitecture == Qwen4ExpArchitecture)
        {
            AddQwen4ExpHyperParameters(g, md, a);
        }

        if (g.Has("rope.freq_base"))
        {
            md[a + "rope.base"] = g.Float("rope.freq_base");
        }

        if (g.Raw("rope.dimension_sections") is Array sections)
        {
            var canonicalSections = new int[RopeSectionCount];
            for (int i = 0; i < Math.Min(RopeSectionCount, sections.Length); i++)
            {
                canonicalSections[i] = Convert.ToInt32(sections.GetValue(i));
            }

            md[a + "rope.sections"] = canonicalSections;
        }

        return md;
    }

    /// <summary>The canonical layer pattern is periodic (every interval-th layer attends); a file whose per-layer flags differ is not supported.</summary>
    private static void CheckRecurrentLayers(GgufMetadataView g, int layerCount, int interval)
    {
        if (g.Raw("attention.recurrent_layers") is not Array flags)
        {
            return;
        }

        for (int il = 0; il < Math.Min(layerCount, flags.Length); il++)
        {
            if (Convert.ToBoolean(flags.GetValue(il)) != ((il + 1) % interval != 0))
            {
                throw new NotSupportedException($"{g.GgufArchitecture}: attention.recurrent_layers is not every {interval}-th layer attention.");
            }
        }
    }

    /// <summary>Hyper-connections, the n-gram embedding (PLE), the QSA indexer and the sigmoid output gate of <c>qwen4exp</c>.</summary>
    private static void AddQwen4ExpHyperParameters(GgufMetadataView g, Dictionary<string, object> md, string a)
    {
        md[a + "hyper_connections.count"] = g.Int("hyper_connection.count");
        md[a + "mtp.layer_count"] = 0;   // the NextN block is not imported (see Split)
        md[a + "linear_attention.gate_activation"] = "sigmoid";
        md[a + "attention.indexer.head_count"] = g.Int("attention.indexer.head_count");
        md[a + "attention.indexer.head_dim"] = g.Int("attention.indexer.key_length");
        md[a + "attention.indexer.token_budget"] = g.Int("attention.indexer.top_k");
        if (g.Raw("attention.compress_ratios") is Array ratios)
        {
            md[a + "attention.indexer.block_size"] = ratios.Cast<object>().Max(Convert.ToInt32);
        }

        if (g.Raw("ple.layers") is not Array { Length: > 0 } pleLayers)
        {
            return;
        }

        if (pleLayers.Length > 1)
        {
            throw new NotSupportedException("qwen4exp: only one n-gram embedding (PLE) layer is supported.");
        }

        md[a + "ngram_embedding.layer"] = Convert.ToInt32(pleLayers.GetValue(0));
        md[a + "ngram_embedding.multipliers"] = LongArray(g, "ple.layer_multipliers");
        md[a + "ngram_embedding.head_sizes"] = LongArray(g, "ple.head_vocab_sizes");
        md[a + "ngram_embedding.head_offsets"] = LongArray(g, "ple.head_offsets");
        md[a + "ngram_embedding.heads_per_order"] = g.Int("ple.heads_per_ngram");
        md[a + "ngram_embedding.stop_token"] = (long)g.Int("ple.eos_token_id");
        md[a + "ngram_embedding.conv_dilation"] = g.Int("ple.ngram_size");
    }

    private static long[] LongArray(GgufMetadataView g, string key) =>
        g.Raw(key) is Array values
            ? [.. values.Cast<object>().Select(v => Convert.ToInt64(v))]
            : throw new InvalidDataException($"GGUF metadata '{key}' is missing or not an array.");

    /// <summary><c>qwen4exp</c> tensors outside the blocks, GGUF name to canonical name (the n-gram table is mapped separately).</summary>
    private static readonly Dictionary<string, string> s_qwen4ExpGlobal = new(StringComparer.Ordinal)
    {
        ["output_hc_norm.weight"] = "output_hc.norm.weight",
        ["output_hc_down.weight"] = "output_hc.down.weight",
        ["output_hc_up.weight"] = "output_hc.up.weight",
    };

    /// <summary>Per-layer tensors (shared by the trunk and the MTP layers), GGUF suffix to canonical suffix.</summary>
    private static readonly Dictionary<string, string> s_layer = new(StringComparer.Ordinal)
    {
        ["attn_norm.weight"] = "input_norm.weight",
        ["post_attention_norm.weight"] = "post_attention_norm.weight",
        ["attn_q.weight"] = "attention.q.weight",
        ["attn_k.weight"] = "attention.k.weight",
        ["attn_v.weight"] = "attention.v.weight",
        ["attn_output.weight"] = "attention.o.weight",
        ["attn_q_norm.weight"] = "attention.q_norm.weight",
        ["attn_k_norm.weight"] = "attention.k_norm.weight",
        ["attn_qkv.weight"] = "linear_attention.qkv.weight",
        ["attn_gate.weight"] = "linear_attention.z.weight",
        ["ssm_beta.weight"] = "linear_attention.beta.weight",
        ["ssm_alpha.weight"] = "linear_attention.alpha.weight",
        ["ssm_out.weight"] = "linear_attention.out.weight",
        ["ssm_norm.weight"] = "linear_attention.norm.weight",
        ["ssm_conv1d.weight"] = "linear_attention.conv1d.weight",
        ["ssm_a"] = "linear_attention.a",
        ["ssm_dt.bias"] = "linear_attention.dt_bias",
        ["ffn_gate.weight"] = "mlp.gate.weight",
        ["ffn_up.weight"] = "mlp.up.weight",
        ["ffn_down.weight"] = "mlp.down.weight",
        ["ffn_gate_inp.weight"] = "mlp.router.weight",
        ["ffn_gate_exps.weight"] = "mlp.experts.gate.weight",
        ["ffn_up_exps.weight"] = "mlp.experts.up.weight",
        ["ffn_gate_up_exps.weight"] = "mlp.experts.gate_up.weight",
        ["ffn_down_exps.weight"] = "mlp.experts.down.weight",
        ["ffn_gate_shexp.weight"] = "mlp.shared.gate.weight",
        ["ffn_up_shexp.weight"] = "mlp.shared.up.weight",
        ["ffn_down_shexp.weight"] = "mlp.shared.down.weight",
        ["ffn_gate_inp_shexp.weight"] = SharedExpertGate,
        // qwen4exp
        ["hc_attn_norm.weight"] = "attention_hc.norm.weight",
        ["hc_attn_down.weight"] = "attention_hc.down.weight",
        ["hc_attn_up.weight"] = "attention_hc.up.weight",
        ["hc_attn_inject.weight"] = "attention_hc.inject.weight",
        ["hc_ffn_norm.weight"] = "ffn_hc.norm.weight",
        ["hc_ffn_down.weight"] = "ffn_hc.down.weight",
        ["hc_ffn_up.weight"] = "ffn_hc.up.weight",
        ["hc_ffn_inject.weight"] = "ffn_hc.inject.weight",
        ["ple_key.weight"] = "ngram.key.weight",
        ["ple_value.weight"] = "ngram.value.weight",
        ["ple_norm_key.weight"] = "ngram.key_norm.weight",
        ["ple_norm_query.weight"] = "ngram.query_norm.weight",
        ["ple_norm_conv.weight"] = "ngram.conv_norm.weight",
        ["ple_conv1d.weight"] = "ngram.conv1d.weight",
        ["indexer.q_proj.weight"] = "attention.indexer.q.weight",
        ["indexer.k_proj.weight"] = "attention.indexer.k.weight",
        ["indexer.q_norm.weight"] = "attention.indexer.q_norm.weight",
        ["indexer.k_norm.weight"] = "attention.indexer.k_norm.weight",
    };

    /// <summary>Tensors shared by all MTP (NextN) layers, GGUF suffix to the canonical name below <c>mtp.</c>.</summary>
    private static readonly Dictionary<string, string> s_nextn = new(StringComparer.Ordinal)
    {
        ["nextn.eh_proj.weight"] = "fc.weight",
        ["nextn.enorm.weight"] = "embedding_norm.weight",
        ["nextn.hnorm.weight"] = "hidden_norm.weight",
        ["nextn.shared_head_norm.weight"] = "output_norm.weight",
        ["nextn.shared_head_head.weight"] = "lm_head.weight",
        ["nextn.embed_tokens.weight"] = "token_embedding.weight",
    };

    private static bool IsQwen3Next(GgufMetadataView g) => g.GgufArchitecture == Qwen3NextArchitecture;

    /// <inheritdoc/>
    public WeightInfo MapWeight(WeightInfo weight, GgufMetadataView gguf)
    {
        // The shared expert's sigmoid gate is a vector in GGUF; canonically it is a [1, hidden] projection.
        if (weight.Name.EndsWith(SharedExpertGate, StringComparison.Ordinal) && weight.Shape.Rank == 1)
        {
            return weight.As(weight.Name, new Shape(1, weight.Shape[0]));
        }

        return weight;
    }

    /// <summary>
    /// Projections that newer llama.cpp converters store as one tensor [a | b] (rows of a, then rows of b): the unfused GGUF names
    /// of both parts and the metadata key that gives the row count of the second part.
    /// </summary>
    private static readonly (string Fused, string First, string Second, string SecondRows)[] s_fused =
    [
        (".attn_qkv_gate.", ".attn_qkv.", ".attn_gate.", "ssm.inner_size"),
        (".ssm_beta_alpha.", ".ssm_beta.", ".ssm_alpha.", "ssm.time_step_rank"),
        (".hc_attn_down_inject.", ".hc_attn_down.", ".hc_attn_inject.", "hyper_connection.count"),
        (".hc_ffn_down_inject.", ".hc_ffn_down.", ".hc_ffn_inject.", "hyper_connection.count"),
    ];

    /// <summary>
    /// Splits fused tensors into their canonical weights: the [a | b] projections of newer converters (<see cref="s_fused"/>) and the
    /// qwen3next <c>ssm_ba</c> layout. The MTP (NextN) block of qwen4exp splits into no weights, since it is not used (as on the
    /// HF path).
    /// </summary>
    public IReadOnlyList<WeightInfo>? Split(string ggufName, WeightInfo tensor, GgufMetadataView gguf)
    {
        if (gguf.GgufArchitecture == Qwen4ExpArchitecture && IsNextnBlock(ggufName, gguf))
        {
            return [];
        }

        foreach (var (fused, first, second, secondRowsKey) in s_fused)
        {
            if (ggufName.Contains(fused, StringComparison.Ordinal))
            {
                long secondRows = gguf.Int(secondRowsKey);
                long firstRows = tensor.Shape[0].Fixed - secondRows;
                return
                [
                    tensor.RowSlice(PartName(ggufName, fused, first, gguf), 0, firstRows),
                    tensor.RowSlice(PartName(ggufName, fused, second, gguf), firstRows, secondRows),
                ];
            }
        }

        return SplitQwen3NextBetaAlpha(ggufName, tensor, gguf);
    }

    private string PartName(string ggufName, string fused, string part, GgufMetadataView gguf)
        => MapWeightName(ggufName.Replace(fused, part, StringComparison.Ordinal), gguf)
            ?? throw new InvalidDataException($"GGUF tensor '{ggufName}': no canonical name for its part '{part.Trim('.')}'.");

    private static bool IsNextnBlock(string ggufName, GgufMetadataView gguf)
        => GgufTensorNames.Block().Match(ggufName) is { Success: true } m
            && int.Parse(m.Groups[1].Value) >= gguf.Int("block_count") - gguf.Int("nextn_predict_layers", 0);

    /// <summary>
    /// qwen3next keeps the HF <c>in_proj_ba</c> layout: per key head [beta (r rows), alpha (r rows)] with r value heads per key
    /// head; it splits into the canonical beta and alpha projections (value heads grouped by key head).
    /// </summary>
    private IReadOnlyList<WeightInfo>? SplitQwen3NextBetaAlpha(string ggufName, WeightInfo tensor, GgufMetadataView gguf)
    {
        if (!ggufName.EndsWith(".ssm_ba.weight", StringComparison.Ordinal)
            || MapWeightName(ggufName.Replace(".ssm_ba.", ".ssm_beta.", StringComparison.Ordinal), gguf) is not { } betaName)
        {
            return null;
        }

        int keyHeads = gguf.Int("ssm.group_count");
        int perKeyHead = gguf.Int("ssm.time_step_rank") / keyHeads;
        var starts = Enumerable.Range(0, keyHeads).Select(h => (long)h * 2 * perKeyHead).ToArray();
        string alphaName = betaName.Replace("linear_attention.beta.", "linear_attention.alpha.", StringComparison.Ordinal);
        return
        [
            tensor.GatherRows(betaName, starts.Select(s => (s, (long)perKeyHead))),
            tensor.GatherRows(alphaName, starts.Select(s => (s + perKeyHead, (long)perKeyHead))),
        ];
    }

    /// <inheritdoc/>
    public string? MapWeightName(string name, GgufMetadataView g)
    {
        if (s_qwen4ExpGlobal.TryGetValue(name, out var mixerName))
        {
            return mixerName;
        }

        // The n-gram (PLE) table is global in GGUF; canonically it belongs to the layer that uses it.
        if (name == "per_layer_token_embd.weight" && g.Raw("ple.layers") is Array { Length: 1 } pleLayers)
        {
            return $"layers.{Convert.ToInt32(pleLayers.GetValue(0))}.ngram.table.weight";
        }

        if (GgufTensorNames.MapGlobal(name) is { } globalName)
        {
            return globalName;
        }

        var m = GgufTensorNames.Block().Match(name);
        if (!m.Success)
        {
            return null;
        }

        int layer = int.Parse(m.Groups[1].Value);
        string suffix = m.Groups[2].Value;
        int trunkLayers = g.Int("block_count") - g.Int("nextn_predict_layers", 0);
        if (layer < trunkLayers)
        {
            return s_layer.TryGetValue(suffix, out var trunkName) ? $"layers.{layer}.{trunkName}" : null;
        }

        // MTP (NextN) layers: per-layer tensors under mtp.layers.{j}, the shared ones under mtp.*.
        if (s_nextn.TryGetValue(suffix, out var sharedName))
        {
            return $"mtp.{sharedName}";
        }

        return s_layer.TryGetValue(suffix, out var mtpName) ? $"mtp.layers.{layer - trunkLayers}.{mtpName}" : null;
    }
}
