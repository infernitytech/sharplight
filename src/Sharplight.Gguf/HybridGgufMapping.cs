using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>granitehybrid</c> (Granite-4.0-H), <c>falcon-h1</c>, <c>mamba2</c>, <c>mamba</c> and <c>jamba</c> (Mamba-1 layers), <c>plamo2</c> (PLaMo-2: its own Mamba layout, fused q/k/v and gate+up, sandwich norms), <c>lfm2</c> (gated short convolutions + attention), <c>lfm2moe</c> (the same with sigmoid-routed experts after the leading dense layers) and <c>lfm3</c> (LFM2 layers with
/// sliding-window attention layers on their own rotary base, partial rotation of the full-attention layers expressed as frequency factors, a per-head
/// sigmoid output gate and optional sandwich norms) → canonical <c>hybrid</c> (documented on <c>HybridAdapter</c>).
/// A layer has attention when its GGUF key/value head count is positive; every layer of these models has the Mamba-2 mixer except the attention-only
/// layers of Granite-4.0-H (Falcon-H1 has both mixers in every layer, recognised by the SSM weights being present).
/// </summary>
internal sealed class HybridGgufMapping : IGgufArchitectureMapping, IGgufSplitMapping
{
    /// <inheritdoc/>
    public IReadOnlyList<string> GgufArchitectures { get; } = ["granitehybrid", "falcon-h1", "mamba2", "lfm2", "lfm2moe", "lfm3", "mamba", "jamba", "plamo2"];
    private static bool IsLfm2(GgufMetadataView g) => g.GgufArchitecture is "lfm2" or "lfm2moe" or "lfm3";
    private static bool IsPlamo2(GgufMetadataView g) => g.GgufArchitecture == "plamo2";

    // llama.cpp expert_gating_func: 1 = softmax, 2 = sigmoid.
    private const int SigmoidGating = 2;
    /// <inheritdoc/>
    public string Architecture => "hybrid";

    /// <summary>The per-layer key/value head counts (a scalar is repeated for every layer).</summary>
    private static int[] KvHeadCounts(GgufMetadataView g, int layerCount)
        => g.Raw("attention.head_count_kv") is Array values && values.Length == layerCount
            ? Enumerable.Range(0, layerCount).Select(i => Convert.ToInt32(values.GetValue(i))).ToArray()
            : Enumerable.Repeat(g.Int("attention.head_count_kv", 0), layerCount).ToArray();

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(GgufMetadataView g)
    {
        string a = Architecture + ".";
        int layerCount = g.Int("block_count");
        int hidden = g.Int("embedding_length");
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = layerCount,
            [a + "hidden_size"] = hidden,
            [a + "norm_eps"] = g.Float("attention.layer_norm_rms_epsilon"),
        };
        if (g.Has("ssm.conv_kernel"))
        {
            md[a + "ssm.conv_kernel"] = g.Int("ssm.conv_kernel");
        }

        if (g.Has("ssm.group_count"))
        {
            // Mamba-2 sizes; Mamba-1 layers take theirs from the weight shapes.
            int ssmHeads = g.Int("ssm.time_step_rank");
            md[a + "ssm.state_size"] = g.Int("ssm.state_size");
            md[a + "ssm.group_count"] = g.Int("ssm.group_count");
            md[a + "ssm.head_count"] = ssmHeads;
            md[a + "ssm.head_dim"] = g.Int("ssm.inner_size") / ssmHeads;
        }

        if (g.Bool("ssm.dt_b_c_rms", false))
        {
            throw new NotSupportedException("mamba: weight-less RMS norms of dt, B and C (Falcon-Mamba) are not supported yet.");
        }

        if (g.Has("shortconv.l_cache"))
        {
            md[a + "shortconv.kernel"] = g.Int("shortconv.l_cache");
        }

        GgufMappingHelpers.AddContextLength(md, a, g);

        int kvHeads = KvHeadCounts(g, layerCount).Max();
        if (kvHeads > 0)
        {
            int headCount = GgufDecoderHyperParameters.LargestOf(g, "attention.head_count");
            md[a + "attention.head_count"] = headCount;
            md[a + "attention.kv_head_count"] = kvHeads;
            md[a + "attention.head_dim"] = g.Int("attention.key_length", hidden / headCount);
            md[a + "attention.scale"] = g.Float("attention.scale", 1f / MathF.Sqrt(g.Int("attention.key_length", hidden / headCount)));
            md[a + "rope.dim"] = g.Int("rope.dimension_count", g.Int("attention.key_length", hidden / headCount));
            md[a + "rope.base"] = g.Float("rope.freq_base", GgufMappingHelpers.DefaultRopeBase);
            // Granite-4.0-H attention is position-less (NoPE) unless the checkpoint is flagged as rope-finetuned; Falcon-H1 and LFM2 always rotate.
            md[a + "attention.rope"] = g.GgufArchitecture is "falcon-h1" or "plamo2" || IsLfm2(g) || g.Bool("rope.scaling.finetuned", false);
        }

        if (g.Int("attention.sliding_window", 0) is > 0 and var window && g.Raw("attention.sliding_window_pattern") is Array pattern)
        {
            md[a + "attention.sliding_window"] = window;
            md[a + "attention.sliding_layers"] = Enumerable.Range(0, layerCount).Select(i => Convert.ToBoolean(pattern.GetValue(i)) ? 1 : 0).ToArray();
            md[a + "rope.local_base"] = g.Float("rope.freq_base_swa", g.Float("rope.freq_base", GgufMappingHelpers.DefaultRopeBase));
        }

        if (g.Has("embedding_scale"))
        {
            md[a + "embedding_scale"] = g.Float("embedding_scale");
        }

        if (g.Has("residual_scale"))
        {
            md[a + "residual_scale"] = g.Float("residual_scale");
        }

        if (g.Has("logit_scale"))
        {
            md[a + "logit_scale"] = 1f / g.Float("logit_scale");   // llama.cpp divides the logits by this value
        }

        if (g.Int("expert_count", 0) > 1)
        {
            md[a + "moe.expert_count"] = g.Int("expert_count");
            md[a + "moe.expert_used_count"] = g.Int("expert_used_count");
            md[a + "moe.normalize_weights"] = g.GgufArchitecture != "jamba";   // Jamba uses the raw softmax probabilities of the selected experts
            if (g.Int("expert_gating_func", 0) == SigmoidGating)
            {
                md[a + "moe.router_activation"] = "sigmoid";
            }

            if (g.Float("expert_weights_scale", 0f) is var scale && scale != 0f)
            {
                md[a + "moe.weights_scale"] = scale;
            }
        }

        return md;
    }

    /// <summary>Per-layer tensors, GGUF suffix to canonical suffix.</summary>
    private static readonly Dictionary<string, string> s_layer = new(StringComparer.Ordinal)
    {
        ["attn_norm.weight"] = "input_norm.weight",
        ["ffn_norm.weight"] = "post_attention_norm.weight",
        ["ffn_norm"] = "post_attention_norm.weight",   // Falcon-H1 omits the suffix
        ["ssm_in.weight"] = "ssm.in.weight",
        ["ssm_out.weight"] = "ssm.out.weight",
        ["ssm_conv1d.weight"] = "ssm.conv1d.weight",
        ["ssm_conv1d.bias"] = "ssm.conv1d.bias",
        ["ssm_dt.bias"] = "ssm.dt_bias",
        ["ssm_a"] = "ssm.a",
        ["ssm_d"] = "ssm.d",
        ["ssm_norm.weight"] = "ssm.norm.weight",
        ["ssm_x.weight"] = "ssm.x.weight",   // PLaMo-2 files: ssm.bcdt.weight (see MapWeightName)
        ["ssm_dt.weight"] = "ssm.dt.weight",
        ["ssm_dt_norm.weight"] = "ssm.dt_norm.weight",
        ["ssm_b_norm.weight"] = "ssm.b_norm.weight",
        ["ssm_c_norm.weight"] = "ssm.c_norm.weight",
        ["shortconv.in_proj.weight"] = "shortconv.in.weight",
        ["shortconv.conv.weight"] = "shortconv.conv.weight",
        ["shortconv.out_proj.weight"] = "shortconv.out.weight",
        ["attn_q_norm.weight"] = "attention.q_norm.weight",
        ["attn_k_norm.weight"] = "attention.k_norm.weight",
        ["attn_q.weight"] = "attention.q.weight",
        ["attn_k.weight"] = "attention.k.weight",
        ["attn_v.weight"] = "attention.v.weight",
        ["attn_output.weight"] = "attention.o.weight",
        ["ffn_gate.weight"] = "mlp.gate.weight",
        ["ffn_up.weight"] = "mlp.up.weight",
        ["ffn_down.weight"] = "mlp.down.weight",
        ["ffn_gate_inp.weight"] = "mlp.router.weight",
        ["exp_probs_b.bias"] = "mlp.router_bias",
        ["ffn_gate_exps.weight"] = "mlp.experts.gate.weight",
        ["ffn_up_exps.weight"] = "mlp.experts.up.weight",
        ["ffn_down_exps.weight"] = "mlp.experts.down.weight",
        ["ffn_gate_shexp.weight"] = "mlp.shared.gate.weight",
        ["ffn_up_shexp.weight"] = "mlp.shared.up.weight",
        ["ffn_down_shexp.weight"] = "mlp.shared.down.weight",
    };

    /// <inheritdoc/>
    public string? MapWeightName(string name, GgufMetadataView g)
    {
        if (name == "token_embd_norm.weight" && IsLfm2(g))
        {
            return "output_norm.weight";   // LFM2 calls its final norm the embedding norm
        }

        if (GgufTensorNames.MapGlobal(name) is { } globalName)
        {
            return globalName;
        }

        if (name == "rope_freqs.weight")
        {
            return "rope.freq_factors";
        }

        var m = GgufTensorNames.Block().Match(name);
        if (!m.Success)
        {
            return null;
        }

        string suffix = m.Groups[2].Value;
        if (g.GgufArchitecture == "lfm3" && s_lfm3Layer.TryGetValue(suffix, out var lfm3Name))
        {
            return $"layers.{m.Groups[1].Value}.{lfm3Name}";
        }

        if (IsPlamo2(g) && s_plamo2Layer.TryGetValue(suffix, out var plamoName))
        {
            return $"layers.{m.Groups[1].Value}.{plamoName}";
        }

        return s_layer.TryGetValue(suffix, out var c) ? $"layers.{m.Groups[1].Value}.{c}" : null;
    }

    /// <summary>LFM3 tensors on top of the shared table: the per-head output gate and the sandwich norms after the mixer and the feed-forward block.</summary>
    private static readonly Dictionary<string, string> s_lfm3Layer = new(StringComparer.Ordinal)
    {
        ["attn_gate.weight"] = "attention.gate.weight",
        ["post_attention_norm.weight"] = "post_mixer_norm.weight",
        ["post_ffw_norm.weight"] = "mlp.out_norm.weight",
    };

    /// <summary>PLaMo-2 tensors that differ from the shared table: the joint (B, C, dt) projection and the sandwich norms around the mixer and the feed-forward block.</summary>
    private static readonly Dictionary<string, string> s_plamo2Layer = new(StringComparer.Ordinal)
    {
        ["ssm_x.weight"] = "ssm.bcdt.weight",
        ["post_attention_norm.weight"] = "post_mixer_norm.weight",
        ["post_attention_norm"] = "post_mixer_norm.weight",
        ["post_ffw_norm.weight"] = "mlp.out_norm.weight",
        ["post_ffw_norm"] = "mlp.out_norm.weight",
        ["ssm_a.weight"] = "ssm.a",
        ["ssm_d.weight"] = "ssm.d",
        ["ssm_dt_norm"] = "ssm.dt_norm.weight",
        ["ssm_b_norm"] = "ssm.b_norm.weight",
        ["ssm_c_norm"] = "ssm.c_norm.weight",
    };

    /// <summary>PLaMo-2 stores the fused q/k/v of an attention layer in one tensor and gate+up (gate rows first) of the feed-forward block in one tensor.</summary>
    public IReadOnlyList<WeightInfo>? Split(string ggufName, WeightInfo tensor, GgufMetadataView gguf)
    {
        var m = GgufTensorNames.Block().Match(ggufName);
        if (!IsPlamo2(gguf) || !m.Success)
        {
            return null;
        }

        int layer = int.Parse(m.Groups[1].Value);
        string prefix = $"layers.{layer}.";
        switch (m.Groups[2].Value)
        {
            case "attn_qkv.weight":
                int heads = PerLayer(gguf, "attention.head_count", layer);
                int kvHeads = PerLayer(gguf, "attention.head_count_kv", layer);
                int headDim = gguf.Int("attention.key_length");
                int valueDim = gguf.Int("attention.value_length", headDim);
                return
                [
                    tensor.RowSlice(prefix + "attention.q.weight", 0, heads * headDim),
                    tensor.RowSlice(prefix + "attention.k.weight", heads * headDim, kvHeads * headDim),
                    tensor.RowSlice(prefix + "attention.v.weight", (heads + kvHeads) * headDim, kvHeads * valueDim),
                ];
            case "ffn_up.weight":
                int ffn = gguf.Int("feed_forward_length");
                return [tensor.RowSlice(prefix + "mlp.gate.weight", 0, ffn), tensor.RowSlice(prefix + "mlp.up.weight", ffn, ffn)];
            default:
                return null;
        }
    }

    private static int PerLayer(GgufMetadataView g, string key, int layer)
        => g.Raw(key) is Array values ? Convert.ToInt32(values.GetValue(layer)) : g.Int(key);

    /// <summary>Mamba-2 A, D are stored [H, 1] and the gated norm [G, inner/G]; canonical weights are flat (Mamba-1 A stays [inner, N]).</summary>
    public WeightInfo MapWeight(WeightInfo w, GgufMetadataView g)
    {
        bool isScalarPerHead = w.Name.EndsWith(".ssm.a", StringComparison.Ordinal) && w.Shape.Rank == 2 && w.Shape[1].Fixed == 1;
        if (isScalarPerHead || w.Name.EndsWith(".ssm.d", StringComparison.Ordinal) || w.Name.EndsWith(".ssm.norm.weight", StringComparison.Ordinal))
        {
            return w.As(w.Name, new Shape(w.Shape.Elements()));
        }

        return w;
    }
}
