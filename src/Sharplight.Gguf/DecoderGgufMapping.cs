using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp decoder-only architectures (<c>llama</c> (Llama, Mistral, SmolLM, ...), <c>qwen2</c>, <c>qwen3</c>,
/// <c>gemma2</c>, <c>gemma3</c>, <c>phi3</c> (fused qkv / gate+up are split), <c>olmoe</c> and <c>qwen3moe</c> (mixture of experts)) → canonical <c>decoder</c> (vocabulary documented on <c>DecoderAdapter</c>).
/// llama.cpp's "normal"-rope checkpoints (<c>llama</c>) store q/k rows permuted for interleaved rotation; they are
/// un-permuted lazily so canonical weights equal those of a Hugging Face import.
/// The Gemma 4 / Muse Glimmer / Flint variants live in <c>DecoderGgufMapping.Extended.cs</c>.
/// </summary>
internal sealed partial class DecoderGgufMapping : IGgufArchitectureMapping, IGgufSplitMapping
{
    /// <inheritdoc/>
    public IReadOnlyList<string> GgufArchitectures { get; } =
    [
        "llama", "qwen2", "qwen3", "gemma2", "gemma3", "phi3", "olmoe", "qwen3moe", "gemma4", "muse-glimmer", "flint",
    ];

    /// <inheritdoc/>
    public string Architecture => "decoder";

    private static bool IsGemma(string architecture) => architecture is "gemma2" or "gemma3";

    /// <summary>Both architectures store q/k rows permuted for llama.cpp's interleaved ("normal") rope.</summary>
    private static bool PermutesQk(string architecture) => architecture is "llama" or "muse-glimmer";

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(GgufMetadataView g)
    {
        string architecture = g.GgufArchitecture;
        switch (architecture)
        {
            case "gemma4":
                return Gemma4Hyper(g);
            case "muse-glimmer":
                return MuseHyper(g);
            case "flint":
                return FlintHyper(g);
        }

        string a = Architecture + ".";
        int embeddingLength = g.Int("embedding_length");
        int headCount = g.Int("attention.head_count");
        int headDim = g.Int("attention.key_length", embeddingLength / headCount);
        var md = GgufDecoderHyperParameters.Common(g, IsGemma(architecture) ? "gelu_tanh" : "silu", "neox", layerNorm: false);

        if (architecture is "olmoe" or "qwen3moe")
        {
            AddMixtureOfExperts(md, a, g, normalizeWeights: architecture == "qwen3moe");
        }
        else if (architecture == "llama" && g.Int("expert_count", 0) > 0)
        {
            AddMixtureOfExperts(md, a, g, normalizeWeights: true);   // Mixtral: experts live under the llama architecture
        }

        if (IsGemma(architecture))
        {
            AddGemma(md, a, g, embeddingLength, headCount, headDim);
        }
        else if (architecture is "llama" or "phi3" && g.Int("attention.sliding_window", 0) is > 0 and var slidingWindow)
        {
            md[a + "attention.sliding_window"] = slidingWindow;   // every layer (Mistral v0.1); absent / 0 = full attention
        }

        AddRopeScaling(md, a, g);
        return md;
    }

    internal static void AddMixtureOfExperts(Dictionary<string, object> md, string a, GgufMetadataView g, bool normalizeWeights)
    {
        md[a + "moe.expert_count"] = g.Int("expert_count");
        md[a + "moe.expert_used_count"] = g.Int("expert_used_count");
        md[a + "moe.ffn_hidden_size"] = g.Int("expert_feed_forward_length", g.Int("feed_forward_length"));
        md[a + "moe.normalize_weights"] = g.Bool("expert_weights_norm", normalizeWeights);
        if (g.Float("expert_weights_scale", 0f) is var scale && scale != 0f)
        {
            md[a + "moe.weights_scale"] = scale;   // llama.cpp: 0 = unscaled
        }

        if (RouterActivation(g.Int("expert_gating_func", 0)) is { } activation)
        {
            md[a + "moe.router_activation"] = activation;
        }

        if (g.Int("expert_group_count", 0) is > 1 and var groups)
        {
            md[a + "moe.group_count"] = groups;
            md[a + "moe.group_used_count"] = g.Int("expert_group_used_count", groups);
        }
    }

    /// <summary>Canonical <c>moe.router_activation</c> of llama.cpp's <c>expert_gating_func</c> (null for 0 = the architecture's default).</summary>
    /// <exception cref="NotSupportedException">The gating function is unknown.</exception>
    internal static string? RouterActivation(int gatingFunction) => gatingFunction switch
    {
        0 => null,
        1 => "softmax",
        2 => "sigmoid",
        3 => "softmax_topk",
        4 => "sqrt_softplus",
        _ => throw new NotSupportedException($"GGUF expert_gating_func {gatingFunction} is not supported."),
    };

    private static void AddGemma(Dictionary<string, object> md, string a, GgufMetadataView g, int embeddingLength, int headCount, int headDim)
    {
        string architecture = g.GgufArchitecture;
        int layerCount = g.Int("block_count");
        md[a + "embedding_scale"] = MathF.Sqrt(embeddingLength);

        // llama.cpp: 1/sqrt(head_dim), except the 27B-class models (n_embd/n_head).
        bool is27BClass = layerCount == (architecture == "gemma2" ? 46 : 62);
        md[a + "attention.scale"] = 1f / MathF.Sqrt(is27BClass ? embeddingLength / (float)headCount : headDim);
        if (g.Has("attn_logit_softcapping"))
        {
            md[a + "attention.logit_softcap"] = g.Float("attn_logit_softcapping");
        }

        if (g.Has("final_logit_softcapping"))
        {
            md[a + "final_logit_softcap"] = g.Float("final_logit_softcapping");
        }

        int window = g.Int("attention.sliding_window", 0);
        if (window <= 0)
        {
            return;
        }

        int pattern = architecture == "gemma2" ? 2 : 6;
        md[a + "attention.sliding_window"] = window;
        md[a + "attention.sliding_layers"] = GgufMappingHelpers.SlidingLayers(layerCount, pattern);
        if (architecture == "gemma3")
        {
            md[a + "rope.local_base"] = g.Float("rope.freq_base_swa", GgufMappingHelpers.DefaultRopeBase);
        }
    }

    internal static void AddRopeScaling(Dictionary<string, object> md, string a, GgufMetadataView g)
    {
        string? scaling = g.Raw("rope.scaling.type") as string;
        if (g.GgufArchitecture == "phi3" || scaling == "longrope")
        {
            // longrope: per-frequency factors come as tensors, the cos/sin magnitude as attn_factor.
            if (g.Has("rope.scaling.attn_factor") && g.Float("rope.scaling.attn_factor") is var attentionFactor && attentionFactor != 1f)
            {
                md[a + "rope.attention_scale"] = attentionFactor;
            }

            if (g.Has("rope.scaling.original_context_length"))
            {
                md[a + "rope.original_context_length"] = g.Int("rope.scaling.original_context_length");
            }
        }
        else if (scaling == "yarn")
        {
            md[a + "rope.attention_scale"] = YarnOf(g).AttentionScale;
        }
        else if (scaling is not (null or "" or "none" or "linear"))
        {
            throw new NotSupportedException($"GGUF rope.scaling.type '{scaling}' is not supported.");
        }
    }

    /// <summary>Per-layer tensors, GGUF suffix to canonical suffix.</summary>
    internal static readonly Dictionary<string, string> LayerTensors = new(StringComparer.Ordinal)
    {
        ["attn_norm.weight"] = "input_norm.weight",
        ["ffn_norm.weight"] = "post_attention_norm.weight",
        ["attn_q.weight"] = "attention.q.weight",
        ["attn_k.weight"] = "attention.k.weight",
        ["attn_v.weight"] = "attention.v.weight",
        ["attn_output.weight"] = "attention.o.weight",
        ["attn_q.bias"] = "attention.q.bias",
        ["attn_k.bias"] = "attention.k.bias",
        ["attn_v.bias"] = "attention.v.bias",
        ["attn_output.bias"] = "attention.o.bias",
        ["attn_q_norm.weight"] = "attention.q_norm.weight",
        ["attn_k_norm.weight"] = "attention.k_norm.weight",
        ["ffn_gate.weight"] = "mlp.gate.weight",
        ["ffn_up.weight"] = "mlp.up.weight",
        ["ffn_down.weight"] = "mlp.down.weight",
        // Layer-norm models and biased MLPs.
        ["attn_norm.bias"] = "input_norm.bias",
        ["ffn_norm.bias"] = "post_attention_norm.bias",
        ["ffn_gate.bias"] = "mlp.gate.bias",
        ["ffn_up.bias"] = "mlp.up.bias",
        ["ffn_down.bias"] = "mlp.down.bias",
        ["ffn_gate_inp.weight"] = "mlp.router.weight",
        ["ffn_gate_exps.weight"] = "mlp.experts.gate.weight",
        ["ffn_up_exps.weight"] = "mlp.experts.up.weight",
        ["ffn_down_exps.weight"] = "mlp.experts.down.weight",
        ["ffn_gate_up_exps.weight"] = "mlp.experts.gate_up.weight",
        // Routing biases: llama.cpp's exp_probs_b is the selection bias, ffn_gate_inp.bias the logit bias (gpt-oss).
        ["exp_probs_b.bias"] = "mlp.router_bias",
        ["ffn_gate_inp.bias"] = "mlp.router.bias",
        ["ffn_gate_exps.bias"] = "mlp.experts.gate.bias",
        ["ffn_up_exps.bias"] = "mlp.experts.up.bias",
        ["ffn_down_exps.bias"] = "mlp.experts.down.bias",
        // BitNet sub-layer norms.
        ["attn_sub_norm.weight"] = "attention.sub_norm.weight",
        ["ffn_sub_norm.weight"] = "mlp.sub_norm.weight",
        // Flint: always-on shared expert and per-head attention sinks.
        ["ffn_gate_shexp.weight"] = "mlp.shared.gate.weight",
        ["ffn_up_shexp.weight"] = "mlp.shared.up.weight",
        ["ffn_down_shexp.weight"] = "mlp.shared.down.weight",
        // Qwen2-MoE: per-token sigmoid gate of the shared expert ([H] in the file, [1, H] canonical).
        ["ffn_gate_inp_shexp.weight"] = "mlp.shared.input_gate.weight",
        ["attn_sinks.weight"] = "attention.sinks",
        // Gemma 2/3: llama.cpp's post_attention_norm / post_ffw_norm are the norms of the sub-layer outputs.
        ["post_attention_norm.weight"] = "attention.out_norm.weight",
        ["post_ffw_norm.weight"] = "mlp.out_norm.weight",
        // Experts in a latent space (Maion-Coder, Muse Spark).
        ["ffn_latent_down.weight"] = "mlp.latent_down.weight",
        ["ffn_latent_up.weight"] = "mlp.latent_up.weight",
    };

    /// <summary>Rope frequency tensors that sit outside the layers.</summary>
    internal static readonly Dictionary<string, string> RopeFactorTensors = new(StringComparer.Ordinal)
    {
        ["rope_freqs.weight"] = "rope.freq_factors",
        ["rope_factors_short.weight"] = "rope.freq_factors",
        ["rope_factors_long.weight"] = "rope.freq_factors_long",
    };

    /// <inheritdoc/>
    public string? MapWeightName(string name, GgufMetadataView g)
    {
        if (GgufTensorNames.MapGlobal(name) is { } globalName)
        {
            return globalName;
        }

        if (RopeFactorTensors.TryGetValue(name, out var ropeName))
        {
            return ropeName;
        }

        bool extended = IsExtended(g.GgufArchitecture);
        if (extended && s_extendedGlobal.TryGetValue(name, out var extendedGlobalName))
        {
            return extendedGlobalName;
        }

        var m = GgufTensorNames.Block().Match(name);
        if (!m.Success)
        {
            return null;
        }

        string layer = m.Groups[1].Value;
        string suffix = m.Groups[2].Value;
        if (extended && s_extended.TryGetValue(suffix, out var extendedName))
        {
            return $"layers.{layer}.{extendedName}";
        }

        return LayerTensors.TryGetValue(suffix, out var canonical) ? $"layers.{layer}.{canonical}" : null;
    }

    /// <summary>Phi-3 stores q/k/v as one tensor and gate/up as one tensor (gate rows first).</summary>
    public IReadOnlyList<WeightInfo>? Split(string name, WeightInfo t, GgufMetadataView g)
    {
        if (g.GgufArchitecture != "phi3")
        {
            return null;
        }

        var m = GgufTensorNames.Block().Match(name);
        if (!m.Success)
        {
            return null;
        }

        string prefix = $"layers.{m.Groups[1].Value}.";
        int headCount = g.Int("attention.head_count");
        int kvHeadCount = g.Int("attention.head_count_kv", headCount);
        int headDim = g.Int("attention.key_length", g.Int("embedding_length") / headCount);
        switch (m.Groups[2].Value)
        {
            case "attn_qkv.weight":
                return
                [
                    t.RowSlice(prefix + "attention.q.weight", 0, headCount * headDim),
                    t.RowSlice(prefix + "attention.k.weight", headCount * headDim, kvHeadCount * headDim),
                    t.RowSlice(prefix + "attention.v.weight", (headCount + kvHeadCount) * headDim, kvHeadCount * headDim),
                ];
            case "ffn_up.weight":
                int ffnLength = g.Int("feed_forward_length");
                if (t.Shape[0].Fixed != 2 * ffnLength)
                {
                    return null;
                }

                return [t.RowSlice(prefix + "mlp.gate.weight", 0, ffnLength), t.RowSlice(prefix + "mlp.up.weight", ffnLength, ffnLength)];
            default:
                return null;
        }
    }

    /// <inheritdoc/>
    public WeightInfo MapWeight(WeightInfo w, GgufMetadataView g)
    {
        if (!PermutesQk(g.GgufArchitecture))
        {
            return w;
        }

        int headCount = g.Int("attention.head_count");
        if (w.Name.EndsWith("attention.q.weight") || w.Name.EndsWith("attention.q.bias"))
        {
            return UnpermuteRows(w, headCount);
        }

        if (w.Name.EndsWith("attention.k.weight") || w.Name.EndsWith("attention.k.bias"))
        {
            return UnpermuteRows(w, g.Int("attention.head_count_kv", headCount));
        }

        return w;
    }

    /// <summary>
    /// Inverts llama.cpp's convert-time permutation (per head: rows [a₀ b₀ a₁ b₁ …] back to [a₀ a₁ … b₀ b₁ …]).
    /// Rows are independent, so this is a byte-level row reorder for every weight type including quantized ones.
    /// </summary>
    internal static WeightInfo UnpermuteRows(WeightInfo w, int headCount)
    {
        long rows = w.Shape[0].Fixed;
        int headRows = checked((int)(rows / headCount));
        int half = headRows / 2;
        return WeightInfo.Derived(w.Name, w.Type, w.Shape, () =>
        {
            var src = w.ReadAll();
            int rowBytes = checked((int)(src.LongLength / rows));
            var dst = new byte[src.Length];
            for (int h = 0; h < headCount; h++)
            {
                for (int r = 0; r < headRows; r++)
                {
                    int from = h * headRows + (r < half ? 2 * r : 2 * (r - half) + 1);
                    Buffer.BlockCopy(src, from * rowBytes, dst, (h * headRows + r) * rowBytes, rowBytes);
                }
            }

            return dst;
        });
    }

    /// <summary>The rotary dimension the metadata implies (RoPE dimension count, else head size).</summary>
    private static int RopeDimension(GgufMetadataView g)
    {
        int headCount = GgufDecoderHyperParameters.LargestOf(g, "attention.head_count");
        return g.Int("rope.dimension_count", g.Int("attention.key_length", g.Int("embedding_length") / headCount));
    }

    // llama.cpp's YaRN magnitude: 0.1·ln(factor) + 1, or with yarn_log_mul (HF mscale_all_dim, mscale 1) the ratio of the two mscales.
    private static RopeScaling.Result YarnOf(GgufMetadataView g)
    {
        float factor = g.Float("rope.scaling.factor");
        float logMul = g.Float("rope.scaling.yarn_log_mul", 0f);
        float? attentionFactor = logMul != 0f ? YarnMscale(factor, 1f) / YarnMscale(factor, logMul) : null;
        return RopeScaling.Yarn(
            RopeDimension(g),
            g.Float("rope.freq_base", GgufMappingHelpers.DefaultRopeBase),
            factor,
            g.Int("rope.scaling.original_context_length"),
            g.Float("rope.scaling.yarn_beta_fast", 32f),
            g.Float("rope.scaling.yarn_beta_slow", 1f),
            attentionFactor);
    }

    private static float YarnMscale(float factor, float mscale) => factor <= 1f ? 1f : 0.1f * mscale * MathF.Log(factor) + 1f;

    /// <inheritdoc/>
    public IEnumerable<WeightInfo> SyntheticWeights(GgufMetadataView g, IReadOnlyDictionary<string, WeightInfo> weights) => RopeFrequencyFactors(g, weights);

    /// <summary>The <c>rope.freq_factors</c> weight that linear / YaRN scaling metadata describes (none when the file stores the factors or scales nothing).</summary>
    internal static IEnumerable<WeightInfo> RopeFrequencyFactors(GgufMetadataView g, IReadOnlyDictionary<string, WeightInfo> weights)
    {
        // Linear / YaRN scaling are described by metadata (llama3 scaling arrives as the stored rope_freqs.weight tensor).
        if (weights.ContainsKey("rope.freq_factors"))
        {
            yield break;
        }

        RopeScaling.Result? scaling = null;
        if (g.Raw("rope.scaling.type") is "linear")
        {
            scaling = RopeScaling.Linear(RopeDimension(g), g.Float("rope.scaling.factor"));
        }
        else if (g.Raw("rope.scaling.type") is "yarn")
        {
            scaling = YarnOf(g);
        }

        if (scaling is null)
        {
            yield break;
        }

        var factors = scaling.Factors;
        yield return WeightInfo.Derived("rope.freq_factors", DataType.F32, new Shape(factors.Length), () =>
        {
            var bytes = new byte[factors.Length * 4];
            Buffer.BlockCopy(factors, 0, bytes, 0, bytes.Length);
            return bytes;
        });
    }
}
