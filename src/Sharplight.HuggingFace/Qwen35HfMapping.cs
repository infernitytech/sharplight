using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.HuggingFace;

/// <summary>
/// Qwen3.5 / Qwen3.6 dense hybrid (HF <c>qwen3_5</c>, <c>qwen3_5_text</c>) → canonical <c>qwen3_5</c>
/// (names and semantics documented on <c>Qwen35Adapter</c>). Weight semantics: norm weights get the
/// implicit +1 folded in (except the linear-attention gated norm), A = -exp(A_log), conv1d as [C, K] f32,
/// and linear-attention V heads reordered from grouped to tiled order when key heads != value heads.
/// The vision tower is mapped in <c>Qwen35HfVision.cs</c>.
/// </summary>
internal sealed partial class Qwen35HfMapping : IHfArchitectureMapping
{
    private const int RopeAxisCount = 4;

    /// <inheritdoc/>
    public string Architecture => "qwen3_5";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text", Qwen3NextModelType, Qwen4ExpModelType, Qwen4ExpTextModelType];

    // ---- hyper-parameters ---------------------------------------------------------------------

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HfImportOptions options)
    {
        int layerCount = c.Int("num_hidden_layers", "n_layers", "n_layer", "num_layers");
        int mtpLayerCount = IncludesMtp(c, options) ? c.GetInt("mtp_num_hidden_layers") ?? 0 : 0;
        int headCount = c.Int("num_attention_heads", "n_head", "n_heads");
        int hidden = c.Int("hidden_size", "n_embd", "dim");
        int headDim = c.GetInt("head_dim") ?? hidden / headCount;
        int interval = c.GetInt("full_attention_interval") ?? 4;
        CheckLayerTypes(c, layerCount, interval);

        string a = Architecture + ".";
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = layerCount,
            [a + "hidden_size"] = hidden,
            [a + "ffn_hidden_size"] = c.GetInt("intermediate_size", "n_inner", "hidden_dim") ?? c.Int("moe_intermediate_size"),
            [a + "norm_eps"] = (float)c.Double("rms_norm_eps", "norm_eps"),
            [a + "attention.head_count"] = headCount,
            [a + "attention.kv_head_count"] = c.Int("num_key_value_heads", "n_kv_heads"),
            [a + "attention.head_dim"] = headDim,
            [a + "full_attention_interval"] = interval,
            [a + "linear_attention.conv_kernel"] = c.Int("linear_conv_kernel_dim"),
            [a + "linear_attention.key_head_dim"] = c.Int("linear_key_head_dim"),
            [a + "linear_attention.key_head_count"] = c.Int("linear_num_key_heads"),
            [a + "linear_attention.value_head_count"] = c.Int("linear_num_value_heads"),
            [a + "linear_attention.value_head_dim"] = c.Int("linear_value_head_dim"),
            [a + "mtp.layer_count"] = mtpLayerCount,
        };
        HfMappingHelpers.AddContextLength(md, a, c, "max_position_embeddings", "n_ctx", "n_positions", "max_length", "max_sequence_length", "model_max_length");

        // Partial rotary: rope dims = head_dim * partial_rotary_factor (default 0.25).
        double partialRotaryFactor = c.RopeDouble("partial_rotary_factor", "partial_rotary_factor", "rope_pct", "rope_percent") ?? 0.25;
        md[a + "rope.dim"] = (int)(headDim * partialRotaryFactor);
        if (c.RopeDouble("rope_theta", "rope_theta", "global_rope_theta", "rope_global_theta", "rotary_emb_base") is { } theta)
        {
            md[a + "rope.base"] = (float)theta;
        }

        if (IsQwen3Next(c))
        {
            // Plain (partial) RoPE, linear-attention value heads stay grouped by key head (no reorder).
            md[a + "linear_attention.grouped_value_heads"] = true;
        }
        else
        {
            md[a + "rope.sections"] = MropeSections(c);
        }

        AddMoeHyperParameters(c, md, a);
        if (IsQwen4Exp(c))
        {
            AddQwen4ExpHyperParameters(c, md, a);
        }

        // TODO: rope scaling (yarn / linear) parameters are not mapped yet; the adapter does not consume them.
        AddVisionHyperParameters(c, md);
        return md;
    }

    /// <summary>M-RoPE sections, padded to 4 axes; Qwen3.5 always uses interleaved M-RoPE, default [11, 11, 10, 0].</summary>
    private static int[] MropeSections(HfConfig c)
    {
        var sections = c.RopeIntArray("mrope_section");
        if (sections == null)
        {
            return [11, 11, 10, 0];
        }

        var padded = new int[RopeAxisCount];
        for (int i = 0; i < Math.Min(RopeAxisCount, sections.Length); i++)
        {
            padded[i] = sections[i];
        }

        return padded;
    }

    private static void CheckLayerTypes(HfConfig c, int layerCount, int interval)
    {
        var types = c.GetStringArray("layer_types");
        if (types == null)
        {
            return;
        }

        if (types.Length != layerCount)
        {
            throw new InvalidDataException($"layer_types has {types.Length} entries, expected num_hidden_layers ({layerCount}).");
        }

        for (int i = 0; i < layerCount; i++)
        {
            bool linear = (i + 1) % interval != 0;
            if ((types[i] == "linear_attention") != linear)
            {
                throw new NotSupportedException($"layer_types[{i}]='{types[i]}' does not follow full_attention_interval={interval}; the canonical qwen3_5 scheme only expresses a fixed interval.");
            }
        }
    }

    // ---- weights ------------------------------------------------------------------------------
    [GeneratedRegex(@"^(?:model\.)?layers\.(\d+)\.(.+)$")]
    private static partial Regex LayerName();

    [GeneratedRegex(@"^mtp\.layers\.(\d+)\.(.+)$")]
    private static partial Regex MtpLayerName();

    /// <summary>The Gated DeltaNet geometry that decides whether and how V heads are reordered.</summary>
    private sealed record LinearAttentionShape(int KeyHeads, int ValueHeads, int KeyHeadDim, int ValueHeadDim, bool Grouped)
    {
        public int ValueHeadsPerKeyHead => ValueHeads / KeyHeads;

        public bool NeedsReorder => !Grouped && KeyHeads != ValueHeads;
    }

    /// <summary>Norm tensors (x * (1 + w)), HF suffix to canonical suffix; the 1 is folded into the weight.</summary>
    private static readonly Dictionary<string, string> s_normPlusOne = new(StringComparer.Ordinal)
    {
        ["input_layernorm.weight"] = "input_norm.weight",
        ["post_attention_layernorm.weight"] = "post_attention_norm.weight",
        ["self_attn.q_norm.weight"] = "attention.q_norm.weight",
        ["self_attn.k_norm.weight"] = "attention.k_norm.weight",
    };

    /// <summary>Tensors that are only renamed, HF suffix to canonical suffix. q_proj rows are already [q|gate] interleaved per head, as the adapter expects.</summary>
    private static readonly Dictionary<string, string> s_plain = new(StringComparer.Ordinal)
    {
        ["self_attn.q_proj.weight"] = "attention.q.weight",
        ["self_attn.k_proj.weight"] = "attention.k.weight",
        ["self_attn.v_proj.weight"] = "attention.v.weight",
        ["self_attn.o_proj.weight"] = "attention.o.weight",
        ["mlp.gate_proj.weight"] = "mlp.gate.weight",
        ["mlp.up_proj.weight"] = "mlp.up.weight",
        ["mlp.down_proj.weight"] = "mlp.down.weight",
        ["linear_attn.norm.weight"] = "linear_attention.norm.weight",   // the gated norm has no +1
    };

    /// <summary>The MTP tensors that are not inside a layer, HF name to canonical name (norms fold the +1).</summary>
    private static readonly Dictionary<string, string> s_mtpNormPlusOne = new(StringComparer.Ordinal)
    {
        ["mtp.pre_fc_norm_embedding.weight"] = "mtp.embedding_norm.weight",
        ["mtp.pre_fc_norm_hidden.weight"] = "mtp.hidden_norm.weight",
        ["mtp.norm.weight"] = "mtp.output_norm.weight",
    };

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HfImportOptions options)
    {
        bool tied = c.GetBool("tie_word_embeddings") ?? false;
        var shape = new LinearAttentionShape(
            c.Int("linear_num_key_heads"),
            c.Int("linear_num_value_heads"),
            c.Int("linear_key_head_dim"),
            c.Int("linear_value_head_dim"),
            Grouped: IsQwen3Next(c));
        if (shape.ValueHeads % shape.KeyHeads != 0)
        {
            throw new InvalidDataException("linear_num_value_heads must be a multiple of linear_num_key_heads.");
        }

        var collector = new HfWeightCollector();
        var targets = new LayerTargets(c, shape, collector, HfPartJoiner.Experts(), HfPartJoiner.RowShards());
        foreach (var (rawName, tensor) in tensors.OrderBy(p => p.Key, StringComparer.Ordinal))
        {
            var mapped = MapTensor(c, options, rawName, tensor, targets, tied);
            if (mapped == TensorOutcome.Unmapped)
            {
                collector.AddUnmapped(rawName);
            }
        }

        collector.ThrowIfUnmapped(c.ModelType);
        targets.Experts.JoinInto(collector, c.GetInt("num_experts") ?? 0);
        targets.TableShards.JoinInto(collector, c.GetInt("split_ngram_parts") ?? 0);
        return collector.Weights;
    }

    private enum TensorOutcome
    {
        /// <summary>Added to the result.</summary>
        Mapped,

        /// <summary>Deliberately not part of this source (derived buffers, tied head, other towers).</summary>
        Skipped,

        /// <summary>No rule handles the tensor.</summary>
        Unmapped,
    }

    /// <summary>
    /// Where layer tensors go: the config, the linear-attention geometry, the weight collector, the per-expert stacker and the
    /// n-gram table shard concatenation.
    /// </summary>
    private sealed record LayerTargets(HfConfig Config, LinearAttentionShape Shape, HfWeightCollector Collector, HfPartJoiner Experts, HfPartJoiner TableShards);

    private static TensorOutcome MapTensor(
        HfConfig c,
        HfImportOptions options,
        string rawName,
        SafetensorsTensor tensor,
        LayerTargets targets,
        bool tied)
    {
        var collector = targets.Collector;
        // Some checkpoints nest the tower below the language model (model.language_model.visual.*).
        rawName = rawName.Replace("model.language_model.visual.", "model.visual.", StringComparison.Ordinal);

        // Vision tower → canonical vision.* weights (Qwen35HfVision.cs); skipped if the config has no vision_config.
        if (IsVisionTensor(rawName))
        {
            if (VisionConfig(c) is null)
            {
                return TensorOutcome.Skipped;
            }

            return Add(collector, MapVisionTensor(rawName, tensor));
        }

        if (rawName.Contains("visual.") || rawName.Contains("vision_") || rawName.Contains("vision.") || rawName.Contains("audio")
            || rawName.Contains("patch_embed") || rawName.Contains("merger."))
        {
            return TensorOutcome.Skipped;
        }

        if (rawName.EndsWith("rotary_emb.inv_freq", StringComparison.Ordinal))
        {
            return TensorOutcome.Skipped;   // derived buffer
        }

        string name = rawName.Replace("language_model.", "");   // llama.cpp: model.language_model.layers.N → model.layers.N

        if (name.StartsWith("mtp.", StringComparison.Ordinal))
        {
            return IncludesMtp(c, options) ? MapMtp(name, tensor, targets) : TensorOutcome.Skipped;
        }

        if (MapOutputMixer(name, tensor.ToWeightInfo()) is { } mixer)
        {
            collector.Add(mixer);
            return TensorOutcome.Mapped;
        }

        switch (name)
        {
            case "model.embed_tokens.weight" or "embed_tokens.weight":
                collector.Add(HfTransforms.Plain(tensor.ToWeightInfo(), "token_embedding.weight"));
                return TensorOutcome.Mapped;
            case "model.norm.weight" or "norm.weight":
                collector.Add(HfTransforms.NormPlusOne(tensor.ToWeightInfo(), "output_norm.weight"));
                return TensorOutcome.Mapped;
            case "lm_head.weight":
                if (!tied)
                {
                    collector.Add(HfTransforms.Plain(tensor.ToWeightInfo(), "lm_head.weight"));
                }

                return TensorOutcome.Mapped;
        }

        var m = LayerName().Match(name);
        if (!m.Success)
        {
            return TensorOutcome.Unmapped;
        }

        return MapLayerTensor($"layers.{m.Groups[1].Value}.", m.Groups[2].Value, tensor, targets);
    }

    /// <summary>Whether the MTP draft layers are imported (not yet for qwen4_exp, whose MTP head has its own stream mixer).</summary>
    private static bool IncludesMtp(HfConfig c, HfImportOptions options) => options.IncludeMtp && !IsQwen4Exp(c);

    private static TensorOutcome Add(HfWeightCollector collector, WeightInfo? weight)
    {
        if (weight is null)
        {
            return TensorOutcome.Unmapped;
        }

        collector.Add(weight);
        return TensorOutcome.Mapped;
    }

    private static TensorOutcome MapMtp(string name, SafetensorsTensor tensor, LayerTargets targets)
    {
        // mtp.layers.{j}.X → mtp.layers.{j}.<canonical X>; fc / norms → mtp.*
        var m = MtpLayerName().Match(name);
        if (m.Success)
        {
            return MapLayerTensor($"mtp.layers.{m.Groups[1].Value}.", m.Groups[2].Value, tensor, targets);
        }

        var w = tensor.ToWeightInfo();
        if (name == "mtp.fc.weight")
        {
            return Add(targets.Collector, HfTransforms.Plain(w, "mtp.fc.weight"));
        }

        return Add(targets.Collector, s_mtpNormPlusOne.TryGetValue(name, out var canonical) ? HfTransforms.NormPlusOne(w, canonical) : null);
    }

    private static TensorOutcome MapLayerTensor(string prefix, string suffix, SafetensorsTensor tensor, LayerTargets targets)
    {
        var w = tensor.ToWeightInfo();
        if (MapFusedLinearAttention(prefix, suffix, w, targets.Shape) is { } parts)
        {
            foreach (var part in parts)
            {
                targets.Collector.Add(part);
            }

            return TensorOutcome.Mapped;
        }

        if (TryMapMoeTensor(prefix, suffix, w, targets))
        {
            return TensorOutcome.Mapped;
        }

        if (IsQwen4Exp(targets.Config) && MapQwen4ExpLayerTensor(prefix, suffix, w, targets) is not TensorOutcome.Unmapped and var outcome)
        {
            return outcome;
        }

        return Add(targets.Collector, MapSingleLayerTensor(prefix, suffix, w, targets.Shape));
    }

    private static WeightInfo? MapSingleLayerTensor(string prefix, string suffix, WeightInfo w, LinearAttentionShape x)
    {
        if (s_normPlusOne.TryGetValue(suffix, out var normName))
        {
            return HfTransforms.NormPlusOne(w, prefix + normName);
        }

        if (s_plain.TryGetValue(suffix, out var plainName))
        {
            return HfTransforms.Plain(w, prefix + plainName);
        }

        // Gated DeltaNet
        switch (suffix)
        {
            case "linear_attn.in_proj_qkv.weight":
                return ReorderRows(w, prefix + "linear_attention.qkv.weight", x, firstRow: 2 * x.KeyHeads * x.KeyHeadDim, headDim: x.ValueHeadDim);
            case "linear_attn.in_proj_z.weight":
                return ReorderRows(w, prefix + "linear_attention.z.weight", x, 0, x.ValueHeadDim);
            case "linear_attn.in_proj_b.weight":
                return ReorderRows(w, prefix + "linear_attention.beta.weight", x, 0, 1);
            case "linear_attn.in_proj_a.weight":
                return ReorderRows(w, prefix + "linear_attention.alpha.weight", x, 0, 1);
            case "linear_attn.out_proj.weight":
                return ReorderCols(w, prefix + "linear_attention.out.weight", x);
            case "linear_attn.conv1d.weight":
                return Conv1d(w, prefix + "linear_attention.conv1d.weight", x);
            case "linear_attn.A_log":
                return PerValueHead(w, prefix + "linear_attention.a", x, static v => -MathF.Exp(v));   // A = -exp(A_log)
            case "linear_attn.dt_bias":
                return PerValueHead(w, prefix + "linear_attention.dt_bias", x, null);
            default:
                return null;
        }
    }

    // ---- V-head reordering (grouped → tiled; identity when num_k_heads == num_v_heads) --------------

    /// <summary>Reorders rows [firstRow, end) of a 2-D weight as V heads of <paramref name="headDim"/> rows each.</summary>
    private static WeightInfo ReorderRows(WeightInfo w, string name, LinearAttentionShape x, int firstRow, int headDim)
    {
        if (!x.NeedsReorder)
        {
            return HfTransforms.Plain(w, name);
        }

        long cols = w.Shape[1].Fixed;
        return WeightInfo.Derived(name, w.Type, w.Shape, () =>
        {
            int elementBytes = w.Type.ElementSize;
            var src = w.ReadAll();
            long rowBytes = cols * elementBytes;
            long offset = firstRow * rowBytes;
            var tail = HfTransforms.ReorderVHeads(src.AsSpan((int)offset).ToArray(), elementBytes, 1, x.KeyHeads, x.ValueHeadsPerKeyHead, headDim * cols);
            Buffer.BlockCopy(tail, 0, src, (int)offset, tail.Length);
            return src;
        });
    }

    /// <summary>out_proj: reorder input columns (V channels) in groups of head_v_dim.</summary>
    private static WeightInfo ReorderCols(WeightInfo w, string name, LinearAttentionShape x)
    {
        if (!x.NeedsReorder)
        {
            return HfTransforms.Plain(w, name);
        }

        long rows = w.Shape[0].Fixed;
        return WeightInfo.Derived(name, w.Type, w.Shape,
            () => HfTransforms.ReorderVHeads(w.ReadAll(), w.Type.ElementSize, rows, x.KeyHeads, x.ValueHeadsPerKeyHead, x.ValueHeadDim));
    }

    /// <summary>1-D per-V-head parameter (A_log, dt_bias): F32, optional element map, V heads reordered.</summary>
    private static WeightInfo PerValueHead(WeightInfo w, string name, LinearAttentionShape x, Func<float, float>? map)
    {
        if (!x.NeedsReorder)
        {
            return map == null ? HfTransforms.Plain(w, name) : HfTransforms.ToF32(w, name, map: map);
        }

        return WeightInfo.Derived(name, DataType.F32, w.Shape, () =>
            HfTransforms.ReorderVHeads(HfTransforms.EncodeF32(HfTransforms.DecodeF32(w, map)), 4, 1, x.KeyHeads, x.ValueHeadsPerKeyHead, 1));
    }

    /// <summary>conv1d [C,1,K] → [C,K] in F32; V channels (after q,k) reordered.</summary>
    private static WeightInfo Conv1d(WeightInfo w, string name, LinearAttentionShape x)
    {
        long channels = w.Shape[0].Fixed;
        long kernel = w.Shape.Elements() / channels;
        var shape = new Shape(channels, kernel);
        if (!x.NeedsReorder)
        {
            return HfTransforms.ToF32(w, name, shape);
        }

        return WeightInfo.Derived(name, DataType.F32, shape, () =>
        {
            var bytes = HfTransforms.EncodeF32(HfTransforms.DecodeF32(w));
            long offset = 2L * x.KeyHeads * x.KeyHeadDim * kernel * 4;
            var tail = HfTransforms.ReorderVHeads(bytes.AsSpan((int)offset).ToArray(), 4, 1, x.KeyHeads, x.ValueHeadsPerKeyHead, x.ValueHeadDim * kernel);
            Buffer.BlockCopy(tail, 0, bytes, (int)offset, tail.Length);
            return bytes;
        });
    }
}
