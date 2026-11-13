using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Gguf;

/// <summary>llama.cpp <c>rwkv7</c> (RWKV-7 G1) → canonical <c>rwkv7</c> (documented on <c>Rwkv7Adapter</c>).</summary>
internal sealed class Rwkv7GgufMapping : IGgufArchitectureMapping
{
    /// <inheritdoc/>
    public IReadOnlyList<string> GgufArchitectures { get; } = ["rwkv7"];
    /// <inheritdoc/>
    public string Architecture => "rwkv7";

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(GgufMetadataView g)
    {
        string a = Architecture + ".";
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = g.Int("block_count"),
            [a + "hidden_size"] = g.Int("embedding_length"),
            [a + "head_size"] = g.Int("wkv.head_size"),
            [a + "norm_eps"] = g.Float("attention.layer_norm_epsilon"),
        };
        GgufMappingHelpers.AddContextLength(md, a, g);
        return md;
    }

    /// <summary>Per-layer tensors, GGUF suffix to canonical suffix.</summary>
    private static readonly Dictionary<string, string> s_layer = new(StringComparer.Ordinal)
    {
        ["attn_norm.weight"] = "input_norm.weight",
        ["attn_norm.bias"] = "input_norm.bias",
        ["attn_norm_2.weight"] = "post_attention_norm.weight",
        ["attn_norm_2.bias"] = "post_attention_norm.bias",
        ["time_mix_lerp_fused.weight"] = "time_mix.lerp.weight",
        ["time_mix_receptance.weight"] = "time_mix.receptance.weight",
        ["time_mix_key.weight"] = "time_mix.key.weight",
        ["time_mix_value.weight"] = "time_mix.value.weight",
        ["time_mix_output.weight"] = "time_mix.output.weight",
        ["time_mix_w0.weight"] = "time_mix.decay_bias",
        ["time_mix_w1.weight"] = "time_mix.decay_down.weight",
        ["time_mix_w2.weight"] = "time_mix.decay_up.weight",
        ["time_mix_a0.weight"] = "time_mix.iclr_bias",
        ["time_mix_a1.weight"] = "time_mix.iclr_down.weight",
        ["time_mix_a2.weight"] = "time_mix.iclr_up.weight",
        ["time_mix_v0.weight"] = "time_mix.value_mix_bias",
        ["time_mix_v1.weight"] = "time_mix.value_mix_down.weight",
        ["time_mix_v2.weight"] = "time_mix.value_mix_up.weight",
        ["time_mix_g1.weight"] = "time_mix.gate_down.weight",
        ["time_mix_g2.weight"] = "time_mix.gate_up.weight",
        ["time_mix_k_k.weight"] = "time_mix.key_norm_scale",
        ["time_mix_k_a.weight"] = "time_mix.key_lr_scale",
        ["time_mix_r_k.weight"] = "time_mix.bonus",
        ["time_mix_ln.weight"] = "time_mix.out_norm.weight",
        ["time_mix_ln.bias"] = "time_mix.out_norm.bias",
        ["channel_mix_lerp_k.weight"] = "channel_mix.lerp.weight",
        ["channel_mix_key.weight"] = "channel_mix.key.weight",
        ["channel_mix_value.weight"] = "channel_mix.value.weight",
    };

    /// <summary>Embedding / output tensors that differ from the shared global names.</summary>
    private static readonly Dictionary<string, string> s_global = new(StringComparer.Ordinal)
    {
        ["token_embd.weight"] = "token_embedding.weight",
        ["token_embd_norm.weight"] = "embedding_norm.weight",
        ["token_embd_norm.bias"] = "embedding_norm.bias",
        ["output_norm.weight"] = "output_norm.weight",
        ["output_norm.bias"] = "output_norm.bias",
        ["output.weight"] = "lm_head.weight",
    };

    /// <inheritdoc/>
    public string? MapWeightName(string name, GgufMetadataView g)
    {
        if (s_global.TryGetValue(name, out var globalName))
        {
            return globalName;
        }

        var m = GgufTensorNames.Block().Match(name);
        return m.Success && s_layer.TryGetValue(m.Groups[2].Value, out var c) ? $"layers.{m.Groups[1].Value}.{c}" : null;
    }

    /// <summary>The fused interpolation factors are stored [6, 1, 1, C]; canonical weights are [6, C].</summary>
    public WeightInfo MapWeight(WeightInfo w, GgufMetadataView g)
        => w.Name.EndsWith(".time_mix.lerp.weight", StringComparison.Ordinal) ? w.As(w.Name, new Shape(w.Shape[0].Fixed, w.Shape.Elements() / w.Shape[0].Fixed)) : w;
}
