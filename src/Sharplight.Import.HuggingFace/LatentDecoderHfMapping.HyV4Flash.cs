using System.Runtime.InteropServices;
using System.Text.Json;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Hunyuan v4 Flash (HF <c>hy_v4_flash</c>, tencent/Hy4-flash; no transformers class, llama.cpp's <c>hy_v4_flash</c> is the reference): the
/// Hunyuan v4 trunk (see the HyV4 part) with sparse and sliding-window MLA layers of one latent geometry (<c>layer_types</c>; the sliding
/// ones have <c>swa_num_attention_heads</c> and their own rope base), sandwich norms (<c>self_attn.post_proj_norm</c>,
/// <c>mlp.post_fc2_norm</c> / <c>mlp.combine_output_norm</c>), latent routed experts (<c>routed_expert_{down,up}_proj</c>), the pooled indexer
/// below <c>self_attn.compressed_indexer.</c> and Engram modules below <c>engram_layer.</c>. Rotary pairs are interleaved, as llama.cpp
/// reads the rope rows. Engram: each module keeps its own table (<c>layers.{i}.engram.table.weight</c>, head offsets within it); the token map
/// (the <c>lookup_table</c> tensor extended by identity to the vocabulary) and the hash multipliers (numpy's generator seeded with
/// <c>engram_seed</c> + 10007 · layer id, bounded by the compressed vocabulary) become the weights <c>engram.token_ids</c> /
/// <c>engram.hash_multipliers</c>. The vision tower (<c>vit.</c>), the dfly draft layers and the MTP block are not imported.
/// </summary>
internal sealed partial class LatentDecoderHfMapping
{
    // The reference leaves the Engram norms' epsilon unset: the fp32 machine epsilon.
    private const float HyV4FlashEngramEps = 1.1920929e-07f;

    // The reference seeds each module's hash multipliers with engram_seed + EngramSeedStride · (1-based layer id).
    private const long EngramSeedStride = 10007;

    private static bool IsHyV4Flash(HfConfig c) => c.ModelType == "hy_v4_flash";

    private static bool IsSparseLayerType(string type) => type is "sparse_attention" or "deepseek_sparse_attention";

    private static string[] HyV4FlashLayerTypes(HfConfig c)
    {
        int layers = c.Int("num_hidden_layers");
        var types = c.GetStringArray("layer_types");
        if (types is null || types.Length < layers || types.Take(layers).Any(t => !IsSparseLayerType(t) && t != "sliding_attention"))
        {
            throw new InvalidDataException("hy_v4_flash: layer_types needs a sparse_attention / sliding_attention entry per layer.");
        }

        return types[..layers];
    }

    private static bool HyV4FlashIndexer(HfConfig c) => (c.GetBool("use_dsa") ?? false) && (c.GetBool("use_compressed_indexer") ?? false);

    // The 1-based Engram layer ids, ascending; none when Engram is off.
    private static int[] HyV4FlashEngramIds(HfConfig c)
        => (c.GetBool("engram_enable") ?? true) && c.Get("engram_layer_ids") is { ValueKind: JsonValueKind.Array } ids
            ? [.. ids.EnumerateArray().Select(e => e.GetInt32()).Order()]
            : [];

    // The bucket counts of every hash head of the module at 1-based layer id, order 2 first.
    private static long[] HyV4FlashEngramPrimes(HfConfig c, int id)
    {
        if (c.Get("engram_layer_head_vocab_sizes") is not { ValueKind: JsonValueKind.Object } sizes
            || !sizes.TryGetProperty(id.ToString(System.Globalization.CultureInfo.InvariantCulture), out var groups))
        {
            throw new InvalidDataException($"hy_v4_flash: engram_layer_head_vocab_sizes has no entry for layer {id}.");
        }

        return [.. groups.EnumerateArray().SelectMany(g => g.EnumerateArray()).Select(p => p.GetInt64())];
    }

    private static IEnumerable<string> ValidateHyV4Flash(HfConfig c)
    {
        if (!(c.GetBool("sandwich_norm") ?? false))
        {
            yield return "hy_v4_flash: only sandwich_norm checkpoints are supported";
        }

        if (c.GetBool("dsa_skip_attn_post_norm") ?? false)
        {
            yield return "hy_v4_flash: dsa_skip_attn_post_norm is not supported";
        }

        foreach (var (sparse, sliding) in new[] { ("qk_nope_head_dim", "swa_qk_nope_head_dim"), ("qk_rope_head_dim", "swa_qk_rope_head_dim"),
            ("v_head_dim", "swa_v_head_dim"), ("kv_lora_rank", "swa_kv_lora_rank") })
        {
            if (c.GetInt(sliding) is { } value && value != c.GetInt(sparse))
            {
                yield return $"hy_v4_flash: the sliding-window layers need the sparse layers' MLA geometry ({sliding} {value} != {sparse})";
            }
        }
    }

    private static void AddHyV4FlashHyperParameters(Dictionary<string, object> md, string a, HfConfig c)
    {
        AddHyV4HyperParameters(md, a, c);
        string[] types = HyV4FlashLayerTypes(c);
        string sparseType = types.FirstOrDefault(IsSparseLayerType) ?? "deepseek_sparse_attention";
        md[a + "rope.style"] = "gptj";
        md[a + "rope.base"] = (float)(c.LayerTypeRopeDouble(sparseType, "rope_theta") ?? 10000.0);
        md[a + "rope.base_sliding"] = (float)(c.LayerTypeRopeDouble("sliding_attention", "rope_theta") ?? 10000.0);
        md[a + "attention.sliding_layers"] = types.Select(t => t == "sliding_attention" ? 1 : 0).ToArray();
        md[a + "attention.sliding_window"] = c.Int("sliding_window");
        md[a + "attention.sliding.head_count"] = c.GetInt("swa_num_attention_heads") ?? c.Int("num_attention_heads");

        if (HyV4FlashIndexer(c))
        {
            md[a + "attention.indexer.rope_style"] = "gptj";
            md[a + "attention.indexer.pool_size"] = c.Int("compressed_indexer_compress_ratio");
            md[a + "attention.indexer.pool_tail"] = true;
            md[a + "attention.indexer.pool_own"] = true;
        }
        else
        {
            md.Remove(a + "attention.indexer.token_budget");   // the sparse layers attend their whole context
        }

        int[] ids = HyV4FlashEngramIds(c);
        if (ids.Length > 0)
        {
            var sizes = new List<long>();
            var offsets = new List<long>();
            foreach (int id in ids)
            {
                long offset = 0;
                foreach (long size in HyV4FlashEngramPrimes(c, id))
                {
                    sizes.Add(size);
                    offsets.Add(offset);   // within the module's own table
                    offset += size;
                }
            }

            md[a + "engram.layers"] = ids.Select(id => id - 1).ToArray();
            md[a + "engram.order"] = c.Int("engram_max_ngram_size");
            md[a + "engram.heads_per_order"] = c.Int("engram_n_head_per_ngram");
            md[a + "engram.head_sizes"] = sizes.ToArray();
            md[a + "engram.head_offsets"] = offsets.ToArray();
            md[a + "engram.pad_token"] = (long)c.Int("engram_pad_id");
            md[a + "engram.norm_eps"] = HyV4FlashEngramEps;
        }
    }

    private static readonly Dictionary<string, string> s_hyV4FlashIndexer = new(StringComparer.Ordinal)
    {
        ["token_key_projection.weight"] = "attention.indexer.k.weight",
        ["token_key_norm.weight"] = "attention.indexer.k_norm.weight",
        ["key_value_projection.weight"] = "attention.indexer.pool_value.weight",
        ["key_gate_projection.weight"] = "attention.indexer.pool_gate.weight",
        ["intra_group_ape"] = "attention.indexer.pool_position.weight",
        ["key_norm.weight"] = "attention.indexer.pool_norm.weight",
        ["query_projection.weight"] = "attention.indexer.q_up.weight",
        ["head_weight_projection.weight"] = "attention.indexer.head_weights.weight",
    };

    private static readonly Dictionary<string, string> s_hyV4FlashLayer = new(StringComparer.Ordinal)
    {
        ["self_attn.post_proj_norm.weight"] = "attention.out_norm.weight",
        ["mlp.post_fc2_norm.weight"] = "mlp.out_norm.weight",
        ["mlp.combine_output_norm.weight"] = "mlp.out_norm.weight",
        ["mlp.routed_expert_down_proj.weight"] = "mlp.latent_down.weight",
        ["mlp.routed_expert_up_proj.weight"] = "mlp.latent_up.weight",
    };

    /// <summary>Whether a tensor outside the text model's layers is one the mapping skips (vision tower, draft layers, MTP block).</summary>
    private static bool IsHyV4FlashSkipped(string name)
        => name.StartsWith("vit.", StringComparison.Ordinal) || name.StartsWith("model.dfly_layers.", StringComparison.Ordinal)
            || name.StartsWith("model.mtp_layers.", StringComparison.Ordinal);

    /// <summary>
    /// Maps a Hunyuan v4 Flash layer tensor that differs from Hunyuan v4's; Engram tensors are collected in <paramref name="engram"/> (they are
    /// fused once all are known). False when the suffix is none of them.
    /// </summary>
    private static bool TryMapHyV4FlashLayer(HfConfig c, int layer, string prefix, string suffix, WeightInfo w, HfWeightCollector collector,
        Dictionary<int, Dictionary<string, WeightInfo>> engram)
    {
        const string Indexer = "self_attn.compressed_indexer.";
        const string Engram = "engram_layer.";
        if (suffix.StartsWith(Indexer, StringComparison.Ordinal))
        {
            if (!HyV4FlashIndexer(c))
            {
                return true;   // without the indexer the sparse layers attend densely
            }

            if (!s_hyV4FlashIndexer.TryGetValue(suffix[Indexer.Length..], out var canonical))
            {
                return false;
            }

            bool plain = canonical.EndsWith("q_up.weight", StringComparison.Ordinal) || canonical.EndsWith("indexer.k.weight", StringComparison.Ordinal)
                || canonical.EndsWith("pool_value.weight", StringComparison.Ordinal) || canonical.EndsWith("pool_gate.weight", StringComparison.Ordinal);
            collector.Add(plain ? HfTransforms.Plain(w, prefix + canonical) : HfTransforms.ToF32(w, prefix + canonical));
            return true;
        }

        if (suffix.StartsWith(Engram, StringComparison.Ordinal))
        {
            if (HyV4FlashEngramIds(c).Length > 0)
            {
                if (!engram.TryGetValue(layer, out var parts))
                {
                    engram[layer] = parts = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
                }

                parts[suffix[Engram.Length..]] = w;
            }

            return true;
        }

        if (suffix == "self_attn.kv_b_proj.weight")
        {
            // The sliding-window layers have more heads over the same latent.
            int heads = HyV4FlashLayerTypes(c)[layer] == "sliding_attention" ? c.GetInt("swa_num_attention_heads") ?? c.Int("num_attention_heads")
                : c.Int("num_attention_heads");
            var (keyUp, valueUp) = LatentAttentionWeights.SplitKeyValueUp(w, heads, NopeDims(c), c.Int("v_head_dim"),
                prefix + "attention.k_up.weight", prefix + "attention.v_up.weight", static t => HfTransforms.DecodeF32(t));
            collector.Add(keyUp);
            collector.Add(valueUp);
            return true;
        }

        if (s_hyV4FlashLayer.TryGetValue(suffix, out var renamed))
        {
            collector.Add(HfTransforms.Plain(w, prefix + renamed));
            return true;
        }

        return false;
    }

    /// <summary>
    /// The fused Engram weights of every module (DeepSeek-V4.1's layout, as llama.cpp's converter writes them), its own table, and once the
    /// token map and the hash multipliers.
    /// </summary>
    /// <exception cref="InvalidDataException">A module misses a tensor.</exception>
    private static void AddHyV4FlashEngram(HfConfig c, Dictionary<int, Dictionary<string, WeightInfo>> engram, HfWeightCollector collector)
    {
        int[] ids = HyV4FlashEngramIds(c);
        if (ids.Length == 0)
        {
            return;
        }

        int streams = c.Int("hc_mult");
        int heads = (c.Int("engram_max_ngram_size") - 1) * c.Int("engram_n_head_per_ngram");
        WeightInfo? lookup = null;
        foreach (int id in ids)
        {
            int layer = id - 1;
            string p = $"layers.{layer}.";
            if (!engram.TryGetValue(layer, out var parts))
            {
                throw new InvalidDataException($"hy_v4_flash: layer {layer} has no engram_layer tensors.");
            }

            WeightInfo Part(string name) => parts.TryGetValue(name, out var t) ? t
                : throw new InvalidDataException($"hy_v4_flash: layer {layer} misses engram_layer.{name}.");
            WeightInfo[] PerStream(string format) => [.. Enumerable.Range(0, streams).Select(s => Part(string.Format(System.Globalization.CultureInfo.InvariantCulture, format, s)))];

            collector.Add(WeightInfo.ConcatRows(p + "engram.kv.weight", [.. PerStream("key_projs.{0}.weight"), Part("value_proj.weight")]));
            collector.Add(HfTransforms.ToF32(WeightInfo.Stack(p + "engram.kv.bias", [.. PerStream("key_projs.{0}.bias"), Part("value_proj.bias")]), p + "engram.kv.bias"));
            var keyNorms = PerStream("norm_key.{0}.weight");
            var queryNorms = PerStream("norm_query.{0}.weight");
            int hidden = (int)keyNorms[0].Shape[0].Fixed;
            collector.Add(WeightInfo.Derived(p + "engram.gate", DataType.F32, new Shape(streams, hidden), () =>
            {
                // The gate folds the key and query norm weights of every stream.
                var gate = new float[streams * hidden];
                for (int s = 0; s < streams; s++)
                {
                    var key = HfTransforms.DecodeF32(keyNorms[s]);
                    var query = HfTransforms.DecodeF32(queryNorms[s]);
                    for (int d = 0; d < hidden; d++)
                    {
                        gate[s * hidden + d] = key[d] * query[d];
                    }
                }

                return HfTransforms.EncodeF32(gate);
            }));
            var lookupNorms = Enumerable.Range(0, heads).Select(h => Part($"lookup_per_head_norms.{h}.weight")).ToArray();
            collector.Add(HfTransforms.ToF32(WeightInfo.Stack(p + "engram.lookup_norm.weight", lookupNorms), p + "engram.lookup_norm.weight"));
            collector.Add(HfTransforms.ToF32(WeightInfo.Stack(p + "engram.hc_scale", PerStream("unified_per_hc_scales.{0}")), p + "engram.hc_scale"));
            collector.Add(HfTransforms.ToF32(Part("unified_pre_gamma"), p + "engram.pre_gamma"));
            collector.Add(HfTransforms.Plain(Part("multi_head_embedding.embedding.weight"), p + "engram.table.weight"));
            lookup ??= parts.TryGetValue("lookup_table", out var table) ? table
                : throw new InvalidDataException("hy_v4_flash: Engram needs the lookup_table tensors (checkpoints with engram_vocab_map.npz are not supported).");
        }

        // The derived weights are read after the config is closed: everything they need is captured here.
        int vocab = c.Int("vocab_size");
        var tokenMap = new Lazy<(int[] Map, long Compressed)>(() => EngramTokenMap(lookup!, vocab));
        collector.Add(WeightInfo.Derived("engram.token_ids", DataType.I32, new Shape(vocab),
            () => MemoryMarshal.AsBytes(tokenMap.Value.Map.AsSpan()).ToArray()));
        int order = c.Int("engram_max_ngram_size");
        long seed = c.GetInt("engram_seed") ?? 0;
        collector.Add(WeightInfo.Derived("engram.hash_multipliers", DataType.I64, new Shape(ids.Length * order), () =>
        {
            // numpy: default_rng(seed + 10007 · id).integers(0, bound, size=order) · 2 + 1 with bound = (int64 max // compressed) // 2.
            long bound = Math.Max(1, long.MaxValue / tokenMap.Value.Compressed / 2);
            var multipliers = ids.SelectMany(id => NumpyRandom.Integers(seed + EngramSeedStride * id, bound, order).Select(m => m * 2 + 1)).ToArray();
            return MemoryMarshal.AsBytes(multipliers.AsSpan()).ToArray();
        }));
    }

    // The token → compressed id map (the lookup table, extended by identity to the vocabulary as the reference does) and the compressed
    // vocabulary size (its largest id + 1).
    private static (int[] Map, long Compressed) EngramTokenMap(WeightInfo lookup, int vocab)
    {
        var bytes = lookup.ReadAll();
        long[] table = lookup.Type == DataType.I64 ? MemoryMarshal.Cast<byte, long>(bytes).ToArray()
            : lookup.Type == DataType.I32 ? [.. MemoryMarshal.Cast<byte, int>(bytes).ToArray().Select(v => (long)v)]
            : throw new InvalidDataException($"hy_v4_flash: engram lookup_table has type {lookup.Type} (i64 or i32).");
        var map = new int[Math.Max(vocab, table.Length)];
        for (int t = 0; t < map.Length; t++)
        {
            map[t] = t < table.Length ? checked((int)table[t]) : t;
        }

        return (map[..vocab], table.Max() + 1);
    }
}
