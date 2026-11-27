using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Hugging Face <c>zaya</c> (ZAYA1, <c>ZayaForCausalLM</c>) → canonical <c>hybrid</c> (documented on <c>HybridAdapter</c>): every layer has a
/// CCA attention mixer (<c>CcaAttention</c>: convolved q/k with residual paths, L2-normalized q/k with a key temperature, delayed values) and a
/// routed MoE block behind the ZAYA MLP router with a skip expert (<c>SkipExpertRouter</c>); per-channel residual scales around both blocks
/// and an input scale on the embeddings.
/// </summary>
/// <remarks>
/// The residual scaling <c>(hs + b_hs)·s_hs + (res + b_res)·s_res</c> is stored as <c>hs·s_hs + res·s_res + (b_hs·s_hs + b_res·s_res)</c>, the
/// two value projections (current, delayed) as one <c>attention.v.weight</c>, the grouped convolution [C, head_dim, 2] as
/// [groups, head_dim, 2·head_dim] (previous-row tap first), all as llama.cpp's converter does.
/// </remarks>
internal sealed partial class ZayaHfMapping : IHfArchitectureMapping
{
    /// <inheritdoc/>
    public string Architecture => "hybrid";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["zaya"];

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig config)
    {
        if ((config.GetInt("num_experts_per_tok") ?? 1) != 1)
        {
            yield return "zaya: only num_experts_per_tok = 1 is supported.";
        }

        if ((config.GetInt("cca_time0") ?? 2) != 2 || (config.GetInt("cca_time1") ?? 2) != 2)
        {
            yield return "zaya: only CCA convolution kernels of 2 (cca_time0, cca_time1) are supported.";
        }

        if (config.GetBool("attention_bias") ?? false)
        {
            yield return "zaya: attention biases are not supported.";
        }

        if (config.GetBool("lm_head_bias") ?? false)
        {
            yield return "zaya: a language-model head bias is not supported.";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        string a = Architecture + ".";
        int layers = c.Int("num_hidden_layers");
        int headDim = c.Int("head_dim");
        double partial = c.LayerTypeRopeDouble("hybrid", "partial_rotary_factor") ?? c.GetDouble("partial_rotary_factor") ?? 1.0;
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = layers,
            [a + "hidden_size"] = c.Int("hidden_size"),
            [a + "norm_eps"] = (float)(c.GetDouble("rms_norm_eps") ?? 1e-5),
            [a + "attention.head_count"] = c.Int("num_attention_heads"),
            [a + "attention.kv_head_count"] = c.Int("num_key_value_heads"),
            [a + "attention.head_dim"] = headDim,
            [a + "attention.scale"] = 1f / MathF.Sqrt(headDim),
            [a + "attention.rope"] = true,
            [a + "rope.base"] = (float)(c.LayerTypeRopeDouble("hybrid", "rope_theta") ?? 5_000_000.0),
            [a + "rope.dim"] = (int)(partial * headDim),
            [a + "mlp.activation"] = c.GetString("hidden_act") ?? "silu",
            [a + "moe.expert_count"] = c.Int("num_experts"),
            [a + "moe.expert_used_count"] = 1,
            [a + "moe.normalize_weights"] = false,
        };
        HfMappingHelpers.AddContextLength(md, a, c);
        if (c.GetStringArray("layer_types") is { } types && types.Contains("hybrid_sliding") && c.GetInt("sliding_window") is { } window)
        {
            md[a + "attention.sliding_window"] = window;
            md[a + "attention.sliding_layers"] = types.Select(t => t == "hybrid_sliding" ? 1 : 0).ToArray();
            md[a + "rope.local_base"] = (float)(c.LayerTypeRopeDouble("hybrid_sliding", "rope_theta") ?? 10_000.0);
        }

        return md;
    }

    [GeneratedRegex(@"^model\.layers\.(\d+)\.(.+)$")]
    private static partial Regex LayerName();

    /// <summary>Per-layer tensors that are only renamed, HF suffix to canonical suffix.</summary>
    private static readonly Dictionary<string, string> s_layer = new(StringComparer.Ordinal)
    {
        ["input_layernorm.weight"] = "input_norm.weight",
        ["post_attention_layernorm.weight"] = "post_attention_norm.weight",
        ["self_attn.qkv_proj.q_proj.weight"] = "attention.q.weight",
        ["self_attn.qkv_proj.k_proj.weight"] = "attention.k.weight",
        ["self_attn.o_proj.weight"] = "attention.o.weight",
        ["self_attn.qkv_proj.conv_qk_depthwise.bias"] = "attention.cca.conv.bias",
        ["self_attn.qkv_proj.conv_qk_grouped.bias"] = "attention.cca.group_conv.bias",
        ["self_attn.qk_norm.temp"] = "attention.cca.key_temperature",
        ["mlp.gate.down_proj.weight"] = "mlp.router.down.weight",
        ["mlp.gate.down_proj.bias"] = "mlp.router.down.bias",
        ["mlp.gate.router_states_scale"] = "mlp.router.state_scale",
        ["mlp.gate.router_mlp.norm.weight"] = "mlp.router.norm.weight",
        ["mlp.gate.router_mlp.fc1.weight"] = "mlp.router.fc1.weight",
        ["mlp.gate.router_mlp.fc1.bias"] = "mlp.router.fc1.bias",
        ["mlp.gate.router_mlp.fc2.weight"] = "mlp.router.fc2.weight",
        ["mlp.gate.router_mlp.fc2.bias"] = "mlp.router.fc2.bias",
        ["mlp.gate.router_mlp.out_proj.weight"] = "mlp.router.out.weight",
        ["mlp.gate.balancing_biases"] = "mlp.router.class_bias",
        ["mlp.experts.gate_up_proj"] = "mlp.experts.gate_up.weight",
        ["mlp.experts.down_proj"] = "mlp.experts.down.weight",
    };

    // The parts of a residual scaling module (post_attention_residual_scale / post_mlp_residual_scale).
    private static readonly string[] s_residualParts = ["hidden_states_scale", "hidden_states_bias", "residual_scale", "residual_bias"];

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        int heads = c.Int("num_attention_heads");
        int kvHeads = c.Int("num_key_value_heads");
        int headDim = c.Int("head_dim");
        var collector = new HfWeightCollector();
        var pending = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
        foreach (var (name, tensor) in tensors.OrderBy(p => p.Key, StringComparer.Ordinal))
        {
            if (name.StartsWith("diffusion_", StringComparison.Ordinal))
            {
                continue;   // the draft heads of the diffusion (TiDAR) decoding, unused by autoregressive decoding
            }

            var w = tensor.ToWeightInfo();
            var m = LayerName().Match(name);
            if (!m.Success)
            {
                if (!TryMapGlobalTensor(c, name, w, collector))
                {
                    collector.AddUnmapped(name);
                }

                continue;
            }

            string prefix = $"layers.{m.Groups[1].Value}.";
            string suffix = m.Groups[2].Value;
            if (s_layer.TryGetValue(suffix, out var renamed))
            {
                collector.Add(HfTransforms.Plain(w, prefix + renamed));
            }
            else if (suffix == "self_attn.qkv_proj.conv_qk_depthwise.weight")
            {
                collector.Add(HfTransforms.ToF32(w, prefix + "attention.cca.conv.weight", new Shape(w.Shape[0].Fixed, w.Shape[2].Fixed)));
            }
            else if (suffix == "self_attn.qkv_proj.conv_qk_grouped.weight")
            {
                // [G·d (out), d (in), 2 (taps)] → [G, d (out), 2·d] with the taps outermost in each row: (previous row, current row).
                long kernel = w.Shape[2].Fixed;
                var rows = HfTransforms.ToF32(w, w.Name, new Shape(w.Shape[0].Fixed, w.Shape[1].Fixed * kernel)).TransposeRowBlocks(w.Name, w.Shape[1].Fixed, kernel);
                collector.Add(rows.As(prefix + "attention.cca.group_conv.weight", new Shape(heads + kvHeads, headDim, headDim * kernel)));
            }
            else if (suffix.StartsWith("self_attn.qkv_proj.v_proj_", StringComparison.Ordinal)
                || suffix.StartsWith("post_attention_residual_scale.", StringComparison.Ordinal)
                || suffix.StartsWith("post_mlp_residual_scale.", StringComparison.Ordinal))
            {
                pending[prefix + suffix] = w;
            }
            else
            {
                collector.AddUnmapped(name);
            }
        }

        for (int il = 0; il < c.Int("num_hidden_layers"); il++)
        {
            string prefix = $"layers.{il}.";
            if (pending.Remove(prefix + "self_attn.qkv_proj.v_proj_current.weight", out var current)
                && pending.Remove(prefix + "self_attn.qkv_proj.v_proj_delayed.weight", out var delayed))
            {
                collector.Add(WeightInfo.ConcatRows(prefix + "attention.v.weight", [current, delayed]));
            }

            AddResidual(pending, prefix, "post_attention_residual_scale.", "residual.mixer_", collector);
            AddResidual(pending, prefix, "post_mlp_residual_scale.", "residual.ffn_", collector);
        }

        foreach (var name in pending.Keys)
        {
            collector.AddUnmapped("model." + name);
        }

        collector.ThrowIfUnmapped("zaya");
        return collector.Weights;
    }

    // (hs + b_hs)·s_hs + (res + b_res)·s_res = hs·s_hs + res·s_res + (b_hs·s_hs + b_res·s_res).
    private static void AddResidual(Dictionary<string, WeightInfo> pending, string prefix, string module, string canonical, HfWeightCollector collector)
    {
        var parts = s_residualParts.Select(part => pending.Remove(prefix + module + part, out var w) ? w : null).ToArray();
        if (parts.Any(p => p is null))
        {
            return;
        }

        var (hsScale, hsBias, resScale, resBias) = (parts[0]!, parts[1]!, parts[2]!, parts[3]!);
        collector.Add(HfTransforms.ToF32(hsScale, prefix + canonical + "out_scale"));
        collector.Add(HfTransforms.ToF32(resScale, prefix + canonical + "in_scale"));
        collector.Add(WeightInfo.Derived(prefix + canonical + "bias", DataType.F32, hsBias.Shape, () =>
        {
            float[] sh = HfTransforms.DecodeF32(hsScale), bh = HfTransforms.DecodeF32(hsBias);
            float[] sr = HfTransforms.DecodeF32(resScale), br = HfTransforms.DecodeF32(resBias);
            var bias = new float[sh.Length];
            for (int i = 0; i < bias.Length; i++)
            {
                bias[i] = bh[i] * sh[i] + br[i] * sr[i];
            }

            return HfTransforms.EncodeF32(bias);
        }));
    }

    private static bool TryMapGlobalTensor(HfConfig c, string name, WeightInfo w, HfWeightCollector collector)
    {
        switch (name)
        {
            case "model.embed_tokens.weight":
                collector.Add(HfTransforms.Plain(w, "token_embedding.weight"));
                return true;
            case "model.norm.weight":
                collector.Add(HfTransforms.Plain(w, "output_norm.weight"));
                return true;
            case "model.input_hidden_states_scale":
                collector.Add(HfTransforms.ToF32(w, "input_scale.weight"));
                return true;
            case "model.input_hidden_states_bias":
                collector.Add(HfTransforms.ToF32(w, "input_scale.bias"));
                return true;
            case "lm_head.weight":
                if (!(c.GetBool("tie_word_embeddings") ?? true))
                {
                    collector.Add(HfTransforms.Plain(w, "lm_head.weight"));
                }

                return true;
            default:
                return false;
        }
    }
}
