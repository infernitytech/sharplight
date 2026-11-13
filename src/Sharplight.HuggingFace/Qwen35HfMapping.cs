using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.HuggingFace;

/// <summary>
/// Qwen3.5 / Qwen3.6 dense hybrid (HF <c>qwen3_5</c>, <c>qwen3_5_text</c>) → canonical <c>qwen3_5</c>
/// (names and semantics documented on <c>Qwen35Adapter</c>). Weight semantics: norm weights get the
/// implicit +1 folded in (except the linear-attention gated norm), A = -exp(A_log), conv1d as [C, K] f32,
/// and linear-attention V heads reordered from grouped to tiled order when key heads != value heads.
/// </summary>
public sealed partial class Qwen35HfMapping : IHfArchitectureMapping
{
    public string Architecture => "qwen3_5";
    public IReadOnlyList<string> ModelTypes { get; } = ["qwen3_5", "qwen3_5_text"];

    // ---- hyper-parameters ---------------------------------------------------------------------
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HfImportOptions options)
    {
        int nLayer = c.Int("num_hidden_layers", "n_layers", "n_layer", "num_layers");
        int nMtp = options.IncludeMtp ? c.GetInt("mtp_num_hidden_layers") ?? 0 : 0;
        int nHead = c.Int("num_attention_heads", "n_head", "n_heads");
        int hidden = c.Int("hidden_size", "n_embd", "dim");
        int headDim = c.GetInt("head_dim") ?? hidden / nHead;
        int interval = c.GetInt("full_attention_interval") ?? 4;
        CheckLayerTypes(c, nLayer, interval);

        string a = Architecture + ".";
        int valueHeads = c.Int("linear_num_value_heads");
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = nLayer,
            [a + "hidden_size"] = hidden,
            [a + "ffn_hidden_size"] = c.Int("intermediate_size", "n_inner", "hidden_dim"),
            [a + "norm_eps"] = (float)c.Double("rms_norm_eps", "norm_eps"),
            [a + "attention.head_count"] = nHead,
            [a + "attention.kv_head_count"] = c.Int("num_key_value_heads", "n_kv_heads"),
            [a + "attention.head_dim"] = headDim,
            [a + "full_attention_interval"] = interval,
            [a + "linear_attention.conv_kernel"] = c.Int("linear_conv_kernel_dim"),
            [a + "linear_attention.key_head_dim"] = c.Int("linear_key_head_dim"),
            [a + "linear_attention.key_head_count"] = c.Int("linear_num_key_heads"),
            [a + "linear_attention.value_head_count"] = valueHeads,
            [a + "linear_attention.value_head_dim"] = c.Int("linear_value_head_dim"),
            [a + "mtp.layer_count"] = nMtp,
        };
        if (c.GetInt("max_position_embeddings", "n_ctx", "n_positions", "max_length", "max_sequence_length", "model_max_length") is { } ctx)
            md[a + "context_length"] = ctx;

        // Partial rotary: rope dims = head_dim * partial_rotary_factor (default 0.25).
        double prf = c.RopeDouble("partial_rotary_factor", "partial_rotary_factor", "rope_pct", "rope_percent") ?? 0.25;
        md[a + "rope.dim"] = (int)(headDim * prf);
        if (c.RopeDouble("rope_theta", "rope_theta", "global_rope_theta", "rope_global_theta", "rotary_emb_base") is { } theta)
            md[a + "rope.base"] = (float)theta;

        // M-RoPE sections, padded to 4 axes; Qwen3.5 always uses interleaved M-RoPE, default [11, 11, 10, 0].
        var sections = c.RopeIntArray("mrope_section");
        int[] four = [0, 0, 0, 0];
        if (sections == null) four = [11, 11, 10, 0];
        else for (int i = 0; i < Math.Min(4, sections.Length); i++) four[i] = sections[i];
        md[a + "rope.sections"] = four;

        // TODO: rope scaling (yarn / linear) parameters are not mapped yet; the adapter does not consume them.
        AddVisionHyperParameters(c, md);
        return md;
    }

    static void CheckLayerTypes(HfConfig c, int nLayer, int interval)
    {
        var types = c.GetStringArray("layer_types");
        if (types == null) return;
        if (types.Length != nLayer) throw new InvalidDataException($"layer_types has {types.Length} entries, expected num_hidden_layers ({nLayer}).");
        for (int i = 0; i < nLayer; i++)
        {
            bool linear = (i + 1) % interval != 0;
            if ((types[i] == "linear_attention") != linear)
                throw new NotSupportedException($"layer_types[{i}]='{types[i]}' does not follow full_attention_interval={interval}; the canonical qwen3_5 scheme only expresses a fixed interval.");
        }
    }

    // ---- weights ------------------------------------------------------------------------------
    [GeneratedRegex(@"^(?:model\.)?layers\.(\d+)\.(.+)$")]
    private static partial Regex LayerName();
    [GeneratedRegex(@"^mtp\.layers\.(\d+)\.(.+)$")]
    private static partial Regex MtpLayerName();

    sealed record Ctx(int NumK, int NumV, int HeadK, int HeadV, int ConvK, int Hidden)
    {
        public int VPerK => NumV / NumK;
        public bool NeedsReorder => NumK != NumV;
    }

    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HfImportOptions options)
    {
        int nLayer = c.Int("num_hidden_layers", "n_layers", "n_layer", "num_layers");
        bool tied = c.GetBool("tie_word_embeddings") ?? false;
        var ctx = new Ctx(c.Int("linear_num_key_heads"), c.Int("linear_num_value_heads"),
                          c.Int("linear_key_head_dim"), c.Int("linear_value_head_dim"),
                          c.Int("linear_conv_kernel_dim"), c.Int("hidden_size", "n_embd", "dim"));
        if (ctx.NumV % ctx.NumK != 0) throw new InvalidDataException("linear_num_value_heads must be a multiple of linear_num_key_heads.");

        var result = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
        var unmapped = new List<string>();

        void Add(WeightInfo w)
        {
            if (!result.TryAdd(w.Name, w)) throw new InvalidDataException($"Two HF tensors map to '{w.Name}'.");
        }

        foreach (var (rawName, t) in tensors.OrderBy(p => p.Key, StringComparer.Ordinal))
        {
            string name = rawName;
            // Vision tower → canonical vision.* weights (Qwen35HfVision.cs); skipped if the config has no vision_config.
            if (IsVisionTensor(name))
            {
                if (VisionConfig(c) is null) continue;
                if (MapVisionTensor(name, t, c) is { } vw) Add(vw); else unmapped.Add(rawName);
                continue;
            }
            if (name.Contains("visual.") || name.Contains("vision_") || name.Contains("vision.") || name.Contains("audio") ||
                name.Contains("patch_embed") || name.Contains("merger."))
                continue;
            if (name.EndsWith("rotary_emb.inv_freq", StringComparison.Ordinal)) continue; // derived buffer

            name = name.Replace("language_model.", "");   // llama.cpp: model.language_model.layers.N → model.layers.N

            if (name.StartsWith("mtp.", StringComparison.Ordinal))
            {
                if (!options.IncludeMtp) continue;
                if (MapMtp(name, t, nLayer, c, ctx) is { } mw) Add(mw); else unmapped.Add(rawName);
                continue;
            }

            switch (name)
            {
                case "model.embed_tokens.weight" or "embed_tokens.weight":
                    Add(HfTransforms.Plain(t.ToWeightInfo(), "token_embedding.weight")); continue;
                case "model.norm.weight" or "norm.weight":
                    Add(HfTransforms.NormPlusOne(t.ToWeightInfo(), "output_norm.weight")); continue;
                case "lm_head.weight":
                    if (!tied) Add(HfTransforms.Plain(t.ToWeightInfo(), "lm_head.weight"));
                    continue;
            }

            var m = LayerName().Match(name);
            if (m.Success && MapLayerTensor($"layers.{m.Groups[1].Value}.", m.Groups[2].Value, t, ctx) is { } w) Add(w);
            else unmapped.Add(rawName);
        }

        if (unmapped.Count > 0)
            throw new NotSupportedException($"qwen3_5: {unmapped.Count} tensor(s) have no canonical mapping: {string.Join(", ", unmapped.Take(10))}{(unmapped.Count > 10 ? ", ..." : "")}");
        return result;
    }

    static WeightInfo? MapMtp(string name, SafetensorsTensor t, int nLayer, HfConfig c, Ctx ctx)
    {
        // mtp.layers.{j}.X → mtp.layers.{j}.<canonical X>; fc / norms → mtp.*
        var m = MtpLayerName().Match(name);
        if (m.Success) return MapLayerTensor($"mtp.layers.{m.Groups[1].Value}.", m.Groups[2].Value, t, ctx);
        var w = t.ToWeightInfo();
        return name switch
        {
            "mtp.fc.weight" => HfTransforms.Plain(w, "mtp.fc.weight"),
            "mtp.pre_fc_norm_embedding.weight" => HfTransforms.NormPlusOne(w, "mtp.embedding_norm.weight"),
            "mtp.pre_fc_norm_hidden.weight" => HfTransforms.NormPlusOne(w, "mtp.hidden_norm.weight"),
            "mtp.norm.weight" => HfTransforms.NormPlusOne(w, "mtp.output_norm.weight"),
            _ => null,
        };
    }

    static WeightInfo? MapLayerTensor(string p, string suffix, SafetensorsTensor t, Ctx x)
    {
        var w = t.ToWeightInfo();
        switch (suffix)
        {
            // norms: x * (1 + w) → fold the 1 (all "*norm.weight" except linear_attn.norm)
            case "input_layernorm.weight": return HfTransforms.NormPlusOne(w, p + "input_norm.weight");
            case "post_attention_layernorm.weight": return HfTransforms.NormPlusOne(w, p + "post_attention_norm.weight");
            case "self_attn.q_norm.weight": return HfTransforms.NormPlusOne(w, p + "attention.q_norm.weight");
            case "self_attn.k_norm.weight": return HfTransforms.NormPlusOne(w, p + "attention.k_norm.weight");

            // full attention (q_proj rows are already [q|gate] interleaved per head, as the adapter expects)
            case "self_attn.q_proj.weight": return HfTransforms.Plain(w, p + "attention.q.weight");
            case "self_attn.k_proj.weight": return HfTransforms.Plain(w, p + "attention.k.weight");
            case "self_attn.v_proj.weight": return HfTransforms.Plain(w, p + "attention.v.weight");
            case "self_attn.o_proj.weight": return HfTransforms.Plain(w, p + "attention.o.weight");

            case "mlp.gate_proj.weight": return HfTransforms.Plain(w, p + "mlp.gate.weight");
            case "mlp.up_proj.weight": return HfTransforms.Plain(w, p + "mlp.up.weight");
            case "mlp.down_proj.weight": return HfTransforms.Plain(w, p + "mlp.down.weight");

            // Gated DeltaNet
            case "linear_attn.in_proj_qkv.weight": return ReorderRows(w, p + "linear_attention.qkv.weight", x, firstRow: 2 * x.NumK * x.HeadK, headDim: x.HeadV);
            case "linear_attn.in_proj_z.weight": return ReorderRows(w, p + "linear_attention.z.weight", x, 0, x.HeadV);
            case "linear_attn.in_proj_b.weight": return ReorderRows(w, p + "linear_attention.beta.weight", x, 0, 1);
            case "linear_attn.in_proj_a.weight": return ReorderRows(w, p + "linear_attention.alpha.weight", x, 0, 1);
            case "linear_attn.out_proj.weight": return ReorderCols(w, p + "linear_attention.out.weight", x);
            case "linear_attn.norm.weight": return HfTransforms.Plain(w, p + "linear_attention.norm.weight"); // no +1
            case "linear_attn.conv1d.weight": return Conv1d(w, p + "linear_attention.conv1d.weight", x);
            case "linear_attn.A_log": return Vec(w, p + "linear_attention.a", x, static v => -MathF.Exp(v));   // A = -exp(A_log)
            case "linear_attn.dt_bias": return Vec(w, p + "linear_attention.dt_bias", x, null);
            default: return null;
        }
    }

    // ---- V-head reordering (grouped → tiled; identity when num_k_heads == num_v_heads) --------------

    static int ElemBytes(WeightInfo w) => w.Type.ElementSize;

    /// <summary>Reorders rows [firstRow, end) of a 2-D weight as V heads of <paramref name="headDim"/> rows each.</summary>
    static WeightInfo ReorderRows(WeightInfo w, string name, Ctx x, int firstRow, int headDim)
    {
        if (!x.NeedsReorder) return HfTransforms.Plain(w, name);
        long cols = w.Shape[1].Fixed;
        return WeightInfo.Derived(name, w.Type, w.Shape, () =>
        {
            int eb = ElemBytes(w);
            var src = w.ReadAll();
            long rowBytes = cols * eb, off = firstRow * rowBytes;
            var tail = HfTransforms.ReorderVHeads(src.AsSpan((int)off).ToArray(), eb, 1, x.NumK, x.VPerK, headDim * cols);
            Buffer.BlockCopy(tail, 0, src, (int)off, tail.Length);
            return src;
        });
    }

    /// <summary>out_proj: reorder input columns (V channels) in groups of head_v_dim.</summary>
    static WeightInfo ReorderCols(WeightInfo w, string name, Ctx x)
    {
        if (!x.NeedsReorder) return HfTransforms.Plain(w, name);
        long rows = w.Shape[0].Fixed;
        return WeightInfo.Derived(name, w.Type, w.Shape,
            () => HfTransforms.ReorderVHeads(w.ReadAll(), ElemBytes(w), rows, x.NumK, x.VPerK, x.HeadV));
    }

    /// <summary>1-D per-V-head parameter (A_log, dt_bias): F32, optional element map, V heads reordered.</summary>
    static WeightInfo Vec(WeightInfo w, string name, Ctx x, Func<float, float>? map)
    {
        if (!x.NeedsReorder) return map == null ? HfTransforms.Plain(w, name) : HfTransforms.ToF32(w, name, map: map);
        return WeightInfo.Derived(name, DataType.F32, w.Shape, () =>
            HfTransforms.ReorderVHeads(HfTransforms.EncodeF32(HfTransforms.DecodeF32(w, map)), 4, 1, x.NumK, x.VPerK, 1));
    }

    /// <summary>conv1d [C,1,K] → [C,K] in F32; V channels (after q,k) reordered.</summary>
    static WeightInfo Conv1d(WeightInfo w, string name, Ctx x)
    {
        long C = w.Shape[0].Fixed, K = w.Shape.Elements() / C;
        var shape = new Shape(C, K);
        if (!x.NeedsReorder) return HfTransforms.ToF32(w, name, shape);
        return WeightInfo.Derived(name, DataType.F32, shape, () =>
        {
            var bytes = HfTransforms.EncodeF32(HfTransforms.DecodeF32(w));
            long off = 2L * x.NumK * x.HeadK * K * 4;
            var tail = HfTransforms.ReorderVHeads(bytes.AsSpan((int)off).ToArray(), 4, 1, x.NumK, x.VPerK, x.HeadV * K);
            Buffer.BlockCopy(tail, 0, bytes, (int)off, tail.Length);
            return bytes;
        });
    }
}
