using Sharplight.Models;

namespace Sharplight.Import.Gguf;

/// <summary>
/// llama.cpp <c>dflash</c> drafters → canonical <c>dflash</c> (vocabulary on <c>DflashDraftHead</c>): block drafters that write the fused
/// features of several target layers (<c>target_layers</c>, the inputs of those layers) into their own KV cache and denoise a block of mask
/// tokens after the pending token in one bidirectional pass. Plain DFlash and DFlash2 (a short per-row convolution around attention and MLP,
/// <c>conv_kernel_size</c> / <c>conv_group_size</c>, and a candidate-lattice selector, <c>selector_rank</c> / <c>selector_top_k</c>) are mapped;
/// the LiLiCorr correction head and the DSpark Markov head (and the DSpark DeepSeek-V4 / Mistral backbones) are rejected by <see cref="Validate"/>.
/// The mask token is the tokenizer's <c>tokenizer.ggml.mask_token_id</c>; drafters without an LM head or token embedding use the target's.
/// </summary>
internal sealed class DflashGgufMapping : IGgufArchitectureMapping
{
    /// <inheritdoc/>
    public IReadOnlyList<string> GgufArchitectures { get; } = ["dflash"];

    /// <inheritdoc/>
    public string Architecture => "dflash";

    /// <inheritdoc/>
    public bool IsDraftModel => true;

    /// <inheritdoc/>
    public IEnumerable<string> Validate(GgufMetadataView g)
    {
        if (g.All.ContainsKey("lilicorr.candidate_topk") || g.Tensor("lilicorr.out_head.weight") is not null)
        {
            yield return "The LiLiCorr correction head of this DFlash drafter is not supported (only plain DFlash and DFlash2 drafters run).";
        }

        if (g.Has("hyper_connection.count") || g.Has("attention.kv_lora_rank") || g.Tensor("markov_w1.weight") is not null)
        {
            yield return "DSpark drafters (Markov head, DeepSeek-V4 or latent-attention stages) are not supported.";
        }

        if (g.Raw("target_layers") is not Array)
        {
            yield return "The DFlash drafter names no target_layers.";
        }

        if (g.All.GetValueOrDefault("tokenizer.ggml.mask_token_id") is null)
        {
            yield return "The DFlash drafter's tokenizer has no mask token (tokenizer.ggml.mask_token_id).";
        }

        if (g.Int("selector_top_k", 0) > 0 && g.Tensor("d2t") is not null)
        {
            yield return "A DFlash2 selector over a reduced draft vocabulary (d2t) is not supported.";
        }

        if (g.Tensor("blk.0.attn_v.weight") is null)
        {
            yield return "DFlash drafters whose values share the key projection (no attn_v) are not supported.";
        }

        if (g.Raw("hidden_activation") is string act && act is not ("silu" or "gelu" or "gelu_pytorch_tanh"))
        {
            yield return $"DFlash hidden activation '{act}' is not supported.";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(GgufMetadataView g)
    {
        string a = Architecture + ".";
        int hidden = g.Int("embedding_length");
        int heads = g.Int("attention.head_count");
        int layers = g.Int("block_count");
        int headDim = g.Int("attention.key_length", hidden / heads);
        int window = g.Int("attention.sliding_window", 0);
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "hidden_size"] = hidden,
            [a + "layer_count"] = layers,
            [a + "head_count"] = heads,
            [a + "kv_head_count"] = g.Int("attention.head_count_kv", heads),
            [a + "head_dim"] = headDim,
            [a + "ffn_hidden_size"] = g.Int("feed_forward_length"),
            [a + "norm_eps"] = g.Float("attention.layer_norm_rms_epsilon"),
            [a + "rope.base"] = g.Float("rope.freq_base", GgufMappingHelpers.DefaultRopeBase),
            [a + "rope.dims"] = g.Int("rope.dimension_count", headDim),
            [a + "target_layers"] = ((Array)(g.Raw("target_layers") ?? throw new InvalidDataException("dflash: target_layers is missing.")))
                .Cast<object>().Select(Convert.ToInt32).ToArray(),
            [a + "block_size"] = g.Int("block_size", 16),
            [a + "causal"] = g.Bool("attention.causal", false),
            [a + "mlp.activation"] = g.Raw("hidden_activation") is string act && act.StartsWith("gelu", StringComparison.Ordinal) ? "gelu_tanh" : "silu",
        };
        if (g.All.GetValueOrDefault("tokenizer.ggml.mask_token_id") is { } mask)
        {
            md[a + "mask_token_id"] = Convert.ToInt32(mask);
        }

        if (window > 0)
        {
            // llama.cpp: sliding_window_pattern[il] = true for sliding layers (every layer when the file has no pattern).
            md[a + "sliding_window"] = window;
            md[a + "sliding_layers"] = g.Raw("attention.sliding_window_pattern") is Array pattern && pattern.Length == layers
                ? Enumerable.Range(0, layers).Select(i => Convert.ToBoolean(pattern.GetValue(i)) ? 1 : 0).ToArray()
                : Enumerable.Repeat(1, layers).ToArray();
        }

        if (g.Int("conv_kernel_size", 0) is > 0 and var kernel && g.Tensor("blk.0.attn_conv_proj.weight") is not null)
        {
            md[a + "conv.kernel_size"] = kernel;
            md[a + "conv.group_size"] = g.Int("conv_group_size");
        }

        if (g.Int("selector_top_k", 0) is > 0 and var topK)
        {
            md[a + "selector.top_k"] = topK;
            md[a + "selector.rank"] = g.Int("selector_rank");
        }

        foreach (var (key, canonical) in s_optionalFloats)
        {
            if (g.Float(key, 0f) is var value && value != 0f)
            {
                md[a + canonical] = value;
            }
        }

        DecoderGgufMapping.AddRopeScaling(md, a, g);
        GgufMappingHelpers.AddContextLength(md, a, g);
        return md;
    }

    // llama.cpp keys whose 0 means "not set".
    private static readonly (string Key, string Canonical)[] s_optionalFloats =
    [
        ("embedding_scale", "embedding_scale"),
        ("attention.scale", "attention.scale"),
        ("attention.value_scale", "attention.value_scale"),
        ("logit_scale", "logit_scale"),
        ("final_logit_softcapping", "final_logit_softcap"),
    ];

    private static readonly Dictionary<string, string> s_global = new(StringComparer.Ordinal)
    {
        ["token_embd.weight"] = "token_embedding.weight",
        ["output_norm.weight"] = "output_norm.weight",
        ["output.weight"] = "lm_head.weight",
        ["rope_freqs.weight"] = "rope.freq_factors",
        ["fc.weight"] = "fc.weight",
        ["enc.output_norm.weight"] = "context_norm.weight",
        ["d2t"] = "draft_to_target",
        ["selector_predecessor.weight"] = "selector.predecessor.weight",
        ["selector_successor.weight"] = "selector.successor.weight",
        ["selector_hidden.weight"] = "selector.hidden.weight",
    };

    private static readonly Dictionary<string, string> s_layer = new(StringComparer.Ordinal)
    {
        ["attn_norm.weight"] = "input_norm.weight",
        ["attn_q.weight"] = "attention.q.weight",
        ["attn_k.weight"] = "attention.k.weight",
        ["attn_v.weight"] = "attention.v.weight",
        ["attn_output.weight"] = "attention.o.weight",
        ["attn_q_norm.weight"] = "attention.q_norm.weight",
        ["attn_k_norm.weight"] = "attention.k_norm.weight",
        ["attn_sinks.weight"] = "attention.sinks",
        ["post_attention_norm.weight"] = "attention.post_norm.weight",
        ["ffn_norm.weight"] = "post_attention_norm.weight",
        ["ffn_gate.weight"] = "mlp.gate.weight",
        ["ffn_up.weight"] = "mlp.up.weight",
        ["ffn_down.weight"] = "mlp.down.weight",
        ["post_ffw_norm.weight"] = "mlp.post_norm.weight",
        ["layer_output_scale.weight"] = "output_scale",
        ["attn_conv_base"] = "attention.conv.base",
        ["attn_conv_proj.weight"] = "attention.conv.proj.weight",
        ["ffn_conv_base"] = "mlp.conv.base",
        ["ffn_conv_proj.weight"] = "mlp.conv.proj.weight",
    };

    /// <inheritdoc/>
    public string? MapWeightName(string name, GgufMetadataView g)
    {
        if (s_global.TryGetValue(name, out var global))
        {
            return global;
        }

        var m = GgufTensorNames.Block().Match(name);
        return m.Success && s_layer.TryGetValue(m.Groups[2].Value, out var layer) ? $"layers.{m.Groups[1].Value}.{layer}" : null;
    }

    /// <inheritdoc/>
    public IEnumerable<WeightInfo> SyntheticWeights(GgufMetadataView g, IReadOnlyDictionary<string, WeightInfo> weights)
        => DecoderGgufMapping.RopeFrequencyFactors(g, weights);
}
