using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>inkling</c> (Thinking Machines Inkling, as the unsloth fork converts it) → canonical <c>hybrid</c> (documented on
/// <c>HybridAdapter</c> and <c>HybridAttention</c>), the same vocabulary as the Hugging Face mapping: attention without rotary embeddings,
/// per-layer key/value head counts (<c>attention.head_count_kv</c> array), sliding-window layers from <c>attention.sliding_window_pattern</c>,
/// q/k norms with scale 1/head_dim, the residual short convolutions <c>shortconv_{k,v,attn,mlp}</c>, the relative position bias
/// (<c>attn_r</c>, <c>attn_rel_proj</c>), the log-floor temperature, dense MLPs in the first <c>dense_block_count</c> layers and sigmoid-routed
/// experts whose <c>expert_shared_count</c> shared experts are scored by the router; logits scaled by 1 / <c>logit_scale_denom</c> and the ids from
/// <c>unpadded_vocab_size</c> on suppressed.
/// </summary>
internal sealed class InklingGgufMapping : IGgufArchitectureMapping
{
    /// <inheritdoc/>
    public IReadOnlyList<string> GgufArchitectures { get; } = ["inkling"];

    /// <inheritdoc/>
    public string Architecture => "hybrid";

    /// <summary>Per-layer tensors, GGUF suffix to canonical suffix.</summary>
    private static readonly Dictionary<string, string> s_layer = new(StringComparer.Ordinal)
    {
        ["attn_norm.weight"] = "input_norm.weight",
        ["ffn_norm.weight"] = "post_attention_norm.weight",
        ["attn_q.weight"] = "attention.q.weight",
        ["attn_k.weight"] = "attention.k.weight",
        ["attn_v.weight"] = "attention.v.weight",
        ["attn_r.weight"] = "attention.relative.weight",
        ["attn_output.weight"] = "attention.o.weight",
        ["attn_q_norm.weight"] = "attention.q_norm.weight",
        ["attn_k_norm.weight"] = "attention.k_norm.weight",
        ["attn_rel_proj.weight"] = "attention.relative_bias.weight",
        ["shortconv_k.weight"] = "attention.k_conv.weight",
        ["shortconv_v.weight"] = "attention.v_conv.weight",
        ["shortconv_attn.weight"] = "post_mixer_conv.weight",
        ["shortconv_mlp.weight"] = "mlp.out_conv.weight",
        ["ffn_gscale.weight"] = "mlp.output_scale",
        ["ffn_gate.weight"] = "mlp.gate.weight",
        ["ffn_up.weight"] = "mlp.up.weight",
        ["ffn_down.weight"] = "mlp.down.weight",
        ["ffn_gate_inp.weight"] = "mlp.router.weight",
        ["exp_probs_b.bias"] = "mlp.router_bias",
        ["ffn_gate_exps.weight"] = "mlp.experts.gate.weight",
        ["ffn_up_exps.weight"] = "mlp.experts.up.weight",
        ["ffn_down_exps.weight"] = "mlp.experts.down.weight",
        ["ffn_gate_shexp.weight"] = "mlp.shared_sinks.gate.weight",
        ["ffn_up_shexp.weight"] = "mlp.shared_sinks.up.weight",
        ["ffn_down_shexp.weight"] = "mlp.shared_sinks.down.weight",
    };

    /// <inheritdoc/>
    public IEnumerable<string> Validate(GgufMetadataView g)
    {
        if (g.Int("expert_gating_func", 2) != 2)
        {
            yield return "inkling: only sigmoid expert gating is supported.";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(GgufMetadataView g)
    {
        string a = Architecture + ".";
        int layers = g.Int("block_count");
        int heads = g.Int("attention.head_count");
        int[] kvHeads = GgufMappingHelpers.PerLayer(g, "attention.head_count_kv", layers, heads);
        int hidden = g.Int("embedding_length");
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = layers,
            [a + "hidden_size"] = hidden,
            [a + "norm_eps"] = g.Float("attention.layer_norm_rms_epsilon"),
            [a + "logit_scale"] = 1f / g.Float("logit_scale_denom", 1f),
            [a + "attention.head_count"] = heads,
            [a + "attention.kv_head_count"] = kvHeads.Max(),
            [a + "attention.layer_kv_head_counts"] = kvHeads,
            [a + "attention.head_dim"] = g.Int("attention.key_length", hidden / heads),
            [a + "attention.scale_by_head_dim"] = true,
            [a + "attention.sliding_window"] = g.Int("attention.sliding_window"),
            [a + "attention.sliding_layers"] = GgufMappingHelpers.SlidingLayers(g, layers, 6, denseFirst: false),
            [a + "mlp.activation"] = "silu",
            [a + "moe.expert_count"] = g.Int("expert_count", 0),
            [a + "moe.expert_used_count"] = g.Int("expert_used_count", 0),
            [a + "moe.router_activation"] = "sigmoid",
            [a + "moe.normalize_weights"] = true,
            [a + "moe.weights_scale"] = g.Float("expert_weights_scale", 1f),
            [a + "moe.shared_sink_count"] = g.Int("expert_shared_count", 0),
        };
        if (g.Int("log_scaling_n_floor", 0) is var floor and > 0)
        {
            md[a + "attention.log_scaling_floor"] = (float)floor;
            md[a + "attention.log_scaling_alpha"] = g.Float("log_scaling_alpha", 0f);
        }

        int vocab = g.Int("vocab_size", 0);
        int unpadded = g.Int("unpadded_vocab_size", vocab);
        if (unpadded > 0 && unpadded < vocab)
        {
            md[a + "logits.suppressed_ranges"] = new[] { unpadded, vocab };
        }

        GgufMappingHelpers.AddContextLength(md, a, g);
        return md;
    }

    /// <inheritdoc/>
    public string? MapWeightName(string name, GgufMetadataView g)
    {
        if (name == "token_embd_norm.weight")
        {
            return "embedding_norm.weight";
        }

        if (GgufTensorNames.MapGlobal(name) is { } globalName)
        {
            return globalName;
        }

        var m = GgufTensorNames.Block().Match(name);
        return m.Success && s_layer.TryGetValue(m.Groups[2].Value, out var canonical) ? $"layers.{m.Groups[1].Value}.{canonical}" : null;
    }

    /// <inheritdoc/>
    public WeightInfo MapWeight(WeightInfo weight, GgufMetadataView g)
    {
        if (!weight.Name.EndsWith(".attention.relative_bias.weight", StringComparison.Ordinal))
        {
            return weight;
        }

        // ggml [E, R] (R rows of E distances) → [E, R]: one row of R mixing weights per backward distance.
        long r = weight.Shape[0].Fixed;
        long extent = weight.Shape[1].Fixed;
        if (weight.Type != DataType.F32)
        {
            throw new InvalidDataException($"inkling: {weight.Name} must be F32 (it is {weight.Type}).");
        }

        return weight.As(weight.Name, new Shape(1, r * extent)).TransposeRowBlocks(weight.Name, r, extent).As(weight.Name, new Shape(extent, r));
    }
}
