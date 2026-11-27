namespace Sharplight.Import.Gguf;

/// <summary>
/// llama.cpp <c>hy_v4_flash</c> (Tencent Hy4-flash): the Hunyuan v4 trunk (hyper-connections without a mixing matrix, expansion weights
/// <c>hyper_connection.magnitude</c>·sigmoid + eps, a learned head; MLA with a sigmoid output gate per value channel and per-head sinks) with
/// sparse and sliding-window layers of one latent geometry (the sliding ones, <c>attention.sliding_window_pattern</c>, have more heads and
/// <c>rope.freq_base_swa</c>), sandwich norms (<c>post_attention_norm</c> after the attention, <c>post_ffw_norm</c> after the feed-forward
/// block), a dense first block and latent MoE layers (<c>ffn_routed_down</c> / <c>ffn_routed_up</c>, SwiGLU clamped on the routed experts only)
/// and DeepSeek-V4.1-style Engram modules (<c>ple.*</c> / <c>engram.*</c> keys) with per-head lookup norms, a key/value bias, a normed value and
/// per-stream offsets of the gate. The sparse layers' pooled indexer caches the value and gate projections of the RMS-normed token key; a pool
/// of <c>attention.indexer.kpool</c> tokens is scored by its gated pooled key, normed (<c>indexer_compressor_norm</c>) and rotated at its first
/// position, and a query attends the best <c>top_k / kpool − 1</c> earlier pools plus its own pool. Rotary pairs are interleaved (the
/// rope rows are kept as the reference stores them). The MTP block and the vision tower are not read (llama.cpp skips them too).
/// </summary>
internal sealed partial class LatentDecoderGgufMapping
{
    // The reference leaves the Engram norms' epsilon unset: the fp32 machine epsilon.
    private const float HyV4FlashEngramEps = 1.1920929e-07f;

    private static bool IsHyV4Flash(GgufMetadataView g) => g.GgufArchitecture == "hy_v4_flash";

    private static void AddHyV4FlashHyperParameters(Dictionary<string, object> md, string a, GgufMetadataView g)
    {
        int layers = TrunkLayers(g);
        float eps = g.Float("hyper_connection.epsilon", 1e-6f);
        md[a + "hyper_connections.streams"] = g.Int("hyper_connection.count");
        md[a + "hyper_connections.eps"] = eps;
        md[a + "hyper_connections.magnitude"] = g.Float("hyper_connection.magnitude", 2f);
        md[a + "hyper_connections.post_offset"] = eps;
        if (g.Has("swiglu_clamp_exp"))
        {
            md[a + "moe.swiglu_clamp"] = GgufMappingHelpers.PerLayerFloat(g, "swiglu_clamp_exp", layers);
            md[a + "moe.swiglu_clamp_input"] = true;   // min(gate, limit) before the SiLU, up clamped to ±limit
        }

        md[a + "attention.sliding_layers"] = Enumerable.Range(0, layers).Select(il => IsSlidingLayer(g, il) ? 1 : 0).ToArray();
        md[a + "attention.sliding_window"] = g.Int("attention.sliding_window");
        md[a + "attention.sliding.head_count"] = HeadCount(g, sliding: true);
        md[a + "rope.base_sliding"] = g.Float("rope.freq_base_swa", g.Float("rope.freq_base", GgufMappingHelpers.DefaultRopeBase));

        md[a + "attention.indexer.rope_style"] = "gptj";
        md[a + "attention.indexer.rope_last"] = true;
        md[a + "attention.indexer.norm_eps"] = g.Float("attention.layer_norm_rms_epsilon");
        md[a + "attention.indexer.pool_size"] = g.Int("attention.indexer.kpool");
        md[a + "attention.indexer.pool_tail"] = true;
        md[a + "attention.indexer.pool_own"] = g.Bool("attention.indexer.kpool_select_tail", false);

        if (g.Raw("ple.layers") is Array { Length: > 0 })
        {
            md[a + "engram.layers"] = MetadataInts(g, "ple.layers");
            md[a + "engram.order"] = g.Int("ple.ngram_size");
            md[a + "engram.heads_per_order"] = g.Int("ple.heads_per_ngram");
            md[a + "engram.multipliers"] = MetadataLongs(g, "ple.layer_multipliers");
            md[a + "engram.head_sizes"] = MetadataLongs(g, "ple.head_vocab_sizes");
            md[a + "engram.head_offsets"] = MetadataLongs(g, "ple.head_offsets");
            md[a + "engram.token_map"] = MetadataInts(g, "engram.token_map");
            md[a + "engram.pad_token"] = (long)g.Int("engram.pad_token_id");
            md[a + "engram.norm_eps"] = HyV4FlashEngramEps;
        }
    }

    private static int[] MetadataInts(GgufMetadataView g, string key) =>
        g.Raw(key) is Array values
            ? [.. values.Cast<object>().Select(Convert.ToInt32)]
            : throw new InvalidDataException($"GGUF metadata '{key}' is missing or not an array.");

    private static long[] MetadataLongs(GgufMetadataView g, string key) =>
        g.Raw(key) is Array values
            ? [.. values.Cast<object>().Select(v => unchecked((long)Convert.ToUInt64(v)))]
            : throw new InvalidDataException($"GGUF metadata '{key}' is missing or not an array.");

    /// <summary>Hunyuan v4 Flash tensors outside the blocks, GGUF name to canonical name.</summary>
    private static readonly Dictionary<string, string> s_hyV4FlashGlobal = new(StringComparer.Ordinal)
    {
        ["output_hc_fn.weight"] = "output_hc.fn.weight",
        ["output_hc_base.weight"] = "output_hc.base",
        ["output_hc_scale.weight"] = "output_hc.scale",
        ["per_layer_token_embd.weight"] = "engram.table.weight",
    };

    /// <summary>Per-layer tensors of Hunyuan v4 Flash beyond the shared latent-decoder ones, GGUF suffix to canonical suffix.</summary>
    private static readonly Dictionary<string, string> s_hyV4Flash = new(StringComparer.Ordinal)
    {
        ["attn_gate.weight"] = "attention.gate.weight",
        ["attn_sinks.weight"] = "attention.sinks",
        ["post_attention_norm.weight"] = "attention.out_norm.weight",
        ["post_ffw_norm.weight"] = "mlp.out_norm.weight",
        ["ffn_routed_down.weight"] = "mlp.latent_down.weight",
        ["ffn_routed_up.weight"] = "mlp.latent_up.weight",
        ["indexer_compressor_value.weight"] = "attention.indexer.pool_value.weight",
        ["indexer_compressor_gate.weight"] = "attention.indexer.pool_gate.weight",
        ["indexer_compressor_ape.weight"] = "attention.indexer.pool_position.weight",
        ["indexer_compressor_norm.weight"] = "attention.indexer.pool_norm.weight",
        ["engram_kv.weight"] = "engram.kv.weight",
        ["engram_kv_b.weight"] = "engram.kv.bias",
        ["engram_gate.weight"] = "engram.gate",
        ["engram_lookup_norm.weight"] = "engram.lookup_norm.weight",
        ["engram_pre_gamma.weight"] = "engram.pre_gamma",
        ["engram_hc_scale.weight"] = "engram.hc_scale",
    };
}
