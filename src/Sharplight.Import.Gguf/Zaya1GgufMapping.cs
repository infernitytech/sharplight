using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.Gguf;

/// <summary>
/// llama.cpp <c>zaya1</c> (Zyphra ZAYA1) → canonical <c>hybrid</c> (documented on <c>HybridAdapter</c>): CCA attention in every layer
/// (<c>ssm_conv1d</c> = the depthwise q/k convolution, <c>cca_conv_grp</c> the grouped one, <c>cca_temp</c> the key temperature, <c>attn_v</c>
/// the current and delayed value halves), the ZAYA MLP router (<c>ffn_gate_inp</c> + bias, <c>ffn_router_*</c>, <c>exp_probs_b</c> over the
/// experts and the skip class) in front of fused gate/up experts, residual scales <c>{attn,ffn}_res_scale_hs</c> /
/// <c>{attn,ffn}_resid_scale</c> (+ bias) and the input scale <c>res_scale_in</c>.
/// </summary>
internal sealed class Zaya1GgufMapping : IGgufArchitectureMapping, IGgufSplitMapping
{
    /// <inheritdoc/>
    public IReadOnlyList<string> GgufArchitectures { get; } = ["zaya1"];

    /// <inheritdoc/>
    public string Architecture => "hybrid";

    /// <inheritdoc/>
    public IEnumerable<string> Validate(GgufMetadataView gguf)
    {
        if (gguf.Int("ssm.conv_kernel", 2) != 2)
        {
            yield return "zaya1: only ssm.conv_kernel = 2 is supported.";
        }

        if (gguf.Int("expert_used_count", 1) != 1)
        {
            yield return "zaya1: only one used expert is supported.";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(GgufMetadataView g)
    {
        string a = Architecture + ".";
        int hidden = g.Int("embedding_length");
        int heads = g.Int("attention.head_count");
        int headDim = g.Int("attention.key_length", hidden / heads);
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = g.Int("block_count"),
            [a + "hidden_size"] = hidden,
            [a + "norm_eps"] = g.Float("attention.layer_norm_rms_epsilon"),
            [a + "attention.head_count"] = heads,
            [a + "attention.kv_head_count"] = g.Int("attention.head_count_kv"),
            [a + "attention.head_dim"] = headDim,
            [a + "attention.scale"] = 1f / MathF.Sqrt(headDim),
            [a + "attention.rope"] = true,
            [a + "rope.dim"] = g.Int("rope.dimension_count", headDim),
            [a + "rope.base"] = g.Float("rope.freq_base", GgufMappingHelpers.DefaultRopeBase),
            [a + "mlp.activation"] = "silu",
            [a + "moe.expert_count"] = g.Int("expert_count"),
            [a + "moe.expert_used_count"] = 1,
            [a + "moe.normalize_weights"] = false,
        };
        GgufMappingHelpers.AddContextLength(md, a, g);
        return md;
    }

    /// <summary>Per-layer tensors, GGUF suffix to canonical suffix.</summary>
    private static readonly Dictionary<string, string> s_layer = new(StringComparer.Ordinal)
    {
        ["attn_norm.weight"] = "input_norm.weight",
        ["ffn_norm.weight"] = "post_attention_norm.weight",
        ["attn_q.weight"] = "attention.q.weight",
        ["attn_k.weight"] = "attention.k.weight",
        ["attn_v.weight"] = "attention.v.weight",
        ["attn_output.weight"] = "attention.o.weight",
        ["ssm_conv1d.weight"] = "attention.cca.conv.weight",
        ["ssm_conv1d.bias"] = "attention.cca.conv.bias",
        ["cca_conv_grp.weight"] = "attention.cca.group_conv.weight",
        ["cca_conv_grp.bias"] = "attention.cca.group_conv.bias",
        ["cca_temp.weight"] = "attention.cca.key_temperature",
        ["attn_res_scale_hs.weight"] = "residual.mixer_out_scale",
        ["attn_resid_scale.weight"] = "residual.mixer_in_scale",
        ["attn_resid_scale.bias"] = "residual.mixer_bias",
        ["ffn_res_scale_hs.weight"] = "residual.ffn_out_scale",
        ["ffn_resid_scale.weight"] = "residual.ffn_in_scale",
        ["ffn_resid_scale.bias"] = "residual.ffn_bias",
        ["ffn_gate_inp.weight"] = "mlp.router.down.weight",
        ["ffn_gate_inp.bias"] = "mlp.router.down.bias",
        ["ffn_router_eda.weight"] = "mlp.router.state_scale",
        ["ffn_router_norm.weight"] = "mlp.router.norm.weight",
        ["ffn_router_fc1.weight"] = "mlp.router.fc1.weight",
        ["ffn_router_fc1.bias"] = "mlp.router.fc1.bias",
        ["ffn_router_fc2.weight"] = "mlp.router.fc2.weight",
        ["ffn_router_fc2.bias"] = "mlp.router.fc2.bias",
        ["ffn_router_out.weight"] = "mlp.router.out.weight",
        ["exp_probs_b.bias"] = "mlp.router.class_bias",
        ["ffn_gate_up_exps.weight"] = "mlp.experts.gate_up.weight",
        ["ffn_down_exps.weight"] = "mlp.experts.down.weight",
    };

    /// <inheritdoc/>
    public string? MapWeightName(string name, GgufMetadataView g)
    {
        switch (name)
        {
            case "res_scale_in.weight":
                return "input_scale.weight";
            case "res_scale_in.bias":
                return "input_scale.bias";
        }

        if (GgufTensorNames.MapGlobal(name) is { } globalName)
        {
            return globalName;
        }

        var m = GgufTensorNames.Block().Match(name);
        return m.Success && s_layer.TryGetValue(m.Groups[2].Value, out var c) ? $"layers.{m.Groups[1].Value}.{c}" : null;
    }

    /// <summary>A fused <c>attn_qkv</c> (the converter's <c>--fuse-qkv</c>) is split into q, k and v.</summary>
    public IReadOnlyList<WeightInfo>? Split(string ggufName, WeightInfo tensor, GgufMetadataView gguf)
    {
        var m = GgufTensorNames.Block().Match(ggufName);
        if (!m.Success || m.Groups[2].Value != "attn_qkv.weight")
        {
            return null;
        }

        string prefix = $"layers.{m.Groups[1].Value}.";
        int heads = gguf.Int("attention.head_count");
        int kvHeads = gguf.Int("attention.head_count_kv");
        int headDim = gguf.Int("attention.key_length", gguf.Int("embedding_length") / heads);
        return
        [
            tensor.RowSlice(prefix + "attention.q.weight", 0, heads * headDim),
            tensor.RowSlice(prefix + "attention.k.weight", heads * headDim, kvHeads * headDim),
            tensor.RowSlice(prefix + "attention.v.weight", (heads + kvHeads) * headDim, kvHeads * headDim),
        ];
    }

    /// <summary>
    /// The grouped convolution is stored [taps, groups, out, in]; the canonical weight is [groups, out, taps·in] (the previous-row tap first),
    /// one matrix per group for an expert-indexed product.
    /// </summary>
    public WeightInfo MapWeight(WeightInfo w, GgufMetadataView g)
    {
        if (!w.Name.EndsWith(".attention.cca.group_conv.weight", StringComparison.Ordinal) || w.Shape.Rank != 4)
        {
            return w;
        }

        if (w.Type.IsQuantized)
        {
            throw new InvalidDataException($"zaya1: {w.Name} must be stored as a float tensor, not {w.Type}.");
        }

        long taps = w.Shape[0].Fixed, groups = w.Shape[1].Fixed, outputs = w.Shape[2].Fixed, inputs = w.Shape[3].Fixed;
        int eb = w.Type.ElementSize;
        return WeightInfo.Derived(w.Name, w.Type, new Shape(groups, outputs, taps * inputs), () =>
        {
            var src = w.ReadAll();
            var dst = new byte[src.Length];
            int rowBytes = checked((int)(inputs * eb));
            for (long k = 0; k < taps; k++)
            {
                for (long gr = 0; gr < groups; gr++)
                {
                    for (long o = 0; o < outputs; o++)
                    {
                        long from = ((k * groups + gr) * outputs + o) * inputs;
                        long to = (gr * outputs + o) * taps * inputs + k * inputs;
                        Buffer.BlockCopy(src, checked((int)(from * eb)), dst, checked((int)(to * eb)), rowBytes);
                    }
                }
            }

            return dst;
        });
    }
}
