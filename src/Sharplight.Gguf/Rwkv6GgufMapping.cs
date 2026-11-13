using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Gguf;

/// <summary>llama.cpp <c>rwkv6</c> (RWKV-6 World / Finch) → canonical <c>rwkv6</c> (documented on <c>Rwkv6Adapter</c>). The stacked interpolation up-projection is split into five weights.</summary>
internal sealed class Rwkv6GgufMapping : IGgufArchitectureMapping, IGgufSplitMapping
{
    private const int Mixes = 5;

    /// <inheritdoc/>
    public IReadOnlyList<string> GgufArchitectures { get; } = ["rwkv6", "rwkv6qwen2"];
    /// <inheritdoc/>
    public string Architecture => "rwkv6";

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(GgufMetadataView g)
    {
        string a = Architecture + ".";
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = g.Int("block_count"),
            [a + "hidden_size"] = g.Int("embedding_length"),
            [a + "head_size"] = g.Int("wkv.head_size"),
            [a + "norm_eps"] = g.Has("attention.layer_norm_rms_epsilon") ? g.Float("attention.layer_norm_rms_epsilon") : g.Float("attention.layer_norm_epsilon"),
            [a + "rescale_every_layers"] = g.Int("rescale_every_n_layers", 0),
            [a + "time_mix_rank"] = g.Int("time_mix_extra_dim"),
        };
        if (g.GgufArchitecture == "rwkv6qwen2")
        {
            md[a + "norm_type"] = "rms";
        }

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
        ["ffn_norm.weight"] = "post_attention_norm.weight",
        ["ffn_gate.weight"] = "mlp.gate.weight",
        ["ffn_up.weight"] = "mlp.up.weight",
        ["ffn_down.weight"] = "mlp.down.weight",
        ["time_mix_receptance.bias"] = "time_mix.receptance.bias",
        ["time_mix_key.bias"] = "time_mix.key.bias",
        ["time_mix_value.bias"] = "time_mix.value.bias",
        ["time_mix_lerp_x.weight"] = "time_mix.lerp_x.weight",
        ["time_mix_lerp_w.weight"] = "time_mix.lerp_w.weight",
        ["time_mix_lerp_k.weight"] = "time_mix.lerp_k.weight",
        ["time_mix_lerp_v.weight"] = "time_mix.lerp_v.weight",
        ["time_mix_lerp_r.weight"] = "time_mix.lerp_r.weight",
        ["time_mix_lerp_g.weight"] = "time_mix.lerp_g.weight",
        ["time_mix_w1.weight"] = "time_mix.mix_down.weight",
        ["time_mix_decay.weight"] = "time_mix.decay_bias",
        ["time_mix_decay_w1.weight"] = "time_mix.decay_down.weight",
        ["time_mix_decay_w2.weight"] = "time_mix.decay_up.weight",
        ["time_mix_first.weight"] = "time_mix.bonus",
        ["time_mix_receptance.weight"] = "time_mix.receptance.weight",
        ["time_mix_key.weight"] = "time_mix.key.weight",
        ["time_mix_value.weight"] = "time_mix.value.weight",
        ["time_mix_gate.weight"] = "time_mix.gate.weight",
        ["time_mix_output.weight"] = "time_mix.output.weight",
        ["time_mix_ln.weight"] = "time_mix.out_norm.weight",
        ["time_mix_ln.bias"] = "time_mix.out_norm.bias",
        ["channel_mix_lerp_k.weight"] = "channel_mix.lerp_k.weight",
        ["channel_mix_lerp_r.weight"] = "channel_mix.lerp_r.weight",
        ["channel_mix_key.weight"] = "channel_mix.key.weight",
        ["channel_mix_receptance.weight"] = "channel_mix.receptance.weight",
        ["channel_mix_value.weight"] = "channel_mix.value.weight",
    };

    private static readonly string[] s_lerpNames = ["w", "k", "v", "r", "g"];

    /// <summary>Embedding / output tensors.</summary>
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

    /// <inheritdoc/>
    public IReadOnlyList<WeightInfo>? Split(string ggufName, WeightInfo tensor, GgufMetadataView gguf)
    {
        var m = GgufTensorNames.Block().Match(ggufName);
        if (!m.Success)
        {
            return null;
        }

        string layer = $"layers.{m.Groups[1].Value}.time_mix.";
        switch (m.Groups[2].Value)
        {
            case "time_mix_w2.weight":
                // [5, C, D]: one [C, D] up-projection per interpolation.
                var shape = new Shape(tensor.Shape[1].Fixed, tensor.Shape[2].Fixed);
                return Enumerable.Range(0, Mixes)
                    .Select(i =>
                    {
                        string name = $"{layer}mix_up.{i}.weight";
                        return tensor.RowSlice(name, i, 1).As(name, shape);
                    })
                    .ToList();
            case "time_mix_lerp_fused.weight":
                // [5, 1, 1, C]: the interpolation factors for w, k, v, r, g.
                var vector = new Shape(tensor.Shape.Elements() / Mixes);
                return s_lerpNames
                    .Select((lerp, i) => tensor.RowSlice($"{layer}lerp_{lerp}.weight", i, 1).As($"{layer}lerp_{lerp}.weight", vector))
                    .ToList();
            default:
                return null;
        }
    }
}
