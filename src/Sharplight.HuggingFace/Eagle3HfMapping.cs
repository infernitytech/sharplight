using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.HuggingFace;

/// <summary>
/// EAGLE-3 drafter checkpoints (EAGLE / SpecForge <c>LlamaForCausalLMEagle3</c>, the <c>midlayer.*</c> layout, and the speculators layout with
/// <c>layers.0.*</c>; <see cref="HfConfig.ModelType"/> reports them as <c>eagle3</c>) → canonical <c>eagle3</c> (vocabulary on
/// <c>Eagle3DraftHead</c>). The fused target layers are <c>eagle_aux_hidden_state_layer_ids</c> (default: 2, the middle one and the third from
/// the end of the target's <c>target_num_hidden_layers</c>); <c>d2t</c> stores offsets, the canonical map absolute target ids.
/// </summary>
internal sealed class Eagle3HfMapping : IHfArchitectureMapping
{
    /// <inheritdoc/>
    public string Architecture => "eagle3";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["eagle3"];

    /// <summary>Whether <paramref name="c"/> describes an EAGLE-3 drafter (llama.cpp's test: a draft vocabulary and a single layer, or the class name).</summary>
    internal static bool IsDrafter(HfConfig c)
        => c.Architectures.Any(a => a.Contains("Eagle3", StringComparison.OrdinalIgnoreCase))
            || (c.Has("draft_vocab_size") && c.GetInt("num_hidden_layers") == 1);

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HfImportOptions options)
    {
        string a = Architecture + ".";
        int hidden = c.GetInt("hidden_size") ?? throw new InvalidDataException("eagle3: hidden_size is missing.");
        int heads = c.GetInt("num_attention_heads") ?? throw new InvalidDataException("eagle3: num_attention_heads is missing.");
        var layers = c.Get("eagle_aux_hidden_state_layer_ids") is { } ids ? ids.EnumerateArray().Select(e => e.GetInt32()).ToArray() : null;
        if (layers is null)
        {
            int target = c.GetInt("target_num_hidden_layers") ?? throw new InvalidDataException("eagle3: eagle_aux_hidden_state_layer_ids or target_num_hidden_layers is needed.");
            layers = [2, target / 2, target - 3];
        }

        return new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "hidden_size"] = hidden,
            [a + "head_count"] = heads,
            [a + "kv_head_count"] = c.GetInt("num_key_value_heads") ?? heads,
            [a + "head_dim"] = c.GetInt("head_dim") ?? hidden / heads,
            [a + "norm_eps"] = (float)(c.GetDouble("rms_norm_eps") ?? 1e-5),
            [a + "rope.base"] = (float)(c.GetDouble("rope_theta") ?? 10000.0),
            [a + "rope.style"] = "neox",
            [a + "target_layers"] = layers,
            [a + "target_hidden_size"] = c.GetInt("target_hidden_size") ?? hidden,
            [a + "norm_before_residual"] = c.GetBool("norm_before_residual") ?? false,
            [a + "norm_before_fc"] = c.GetBool("norm_before_fc") ?? false,
            [a + "context_length"] = c.GetInt("max_position_embeddings") ?? 4096,
        };
    }

    private static readonly Dictionary<string, string> s_layer = new(StringComparer.Ordinal)
    {
        ["input_layernorm.weight"] = "input_norm.weight",
        ["hidden_norm.weight"] = "hidden_norm.weight",
        ["self_attn.q_proj.weight"] = "attention.q.weight",
        ["self_attn.k_proj.weight"] = "attention.k.weight",
        ["self_attn.v_proj.weight"] = "attention.v.weight",
        ["self_attn.o_proj.weight"] = "attention.o.weight",
        ["post_attention_layernorm.weight"] = "post_attention_norm.weight",
        ["mlp.gate_proj.weight"] = "mlp.gate.weight",
        ["mlp.up_proj.weight"] = "mlp.up.weight",
        ["mlp.down_proj.weight"] = "mlp.down.weight",
    };

    private static readonly Dictionary<string, string> s_global = new(StringComparer.Ordinal)
    {
        ["fc.weight"] = "fc.weight",
        ["input_norm.weight"] = "fc_norm.weight",
        ["norm.weight"] = "output_norm.weight",
        ["lm_head.weight"] = "lm_head.weight",
        ["embed_tokens.weight"] = "token_embedding.weight",
    };

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HfImportOptions options)
    {
        var weights = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
        foreach (var (name, tensor) in tensors)
        {
            string local = name.StartsWith("model.", StringComparison.Ordinal) ? name["model.".Length..] : name;
            string? canonical = null;
            foreach (var prefix in (string[])["midlayer.", "layers.0."])
            {
                if (local.StartsWith(prefix, StringComparison.Ordinal) && s_layer.TryGetValue(local[prefix.Length..], out var layer))
                {
                    canonical = "layers.0." + layer;
                }
            }

            canonical ??= s_global.GetValueOrDefault(local);
            if (local == "d2t")
            {
                weights["draft_to_target"] = AbsoluteTargetIds(tensor.ToWeightInfo());
                continue;
            }

            if (local == "t2d")
            {
                continue;   // the inverse map, not needed
            }

            weights[canonical ?? throw new InvalidDataException($"eagle3: unexpected tensor '{name}'.")] = tensor.ToWeightInfo().As(canonical);
        }

        var scaling = DecoderHfMapping.ComputeScaling(c, c.GetInt("head_dim") ?? c.GetInt("hidden_size")!.Value / c.GetInt("num_attention_heads")!.Value,
            (float)(c.GetDouble("rope_theta") ?? 10000.0));
        if (scaling is not null)
        {
            weights["rope.freq_factors"] = HfMappingHelpers.F32Weight("rope.freq_factors", scaling.Factors);
        }

        return weights;
    }

    // d2t[i] is the offset of draft token i's target id from i.
    private static WeightInfo AbsoluteTargetIds(WeightInfo offsets)
    {
        int n = (int)offsets.Shape.Elements();
        return WeightInfo.Derived("draft_to_target", DataType.I32, new Shape(n), () =>
        {
            var raw = offsets.ReadAll();
            var bytes = new byte[n * 4];
            for (int i = 0; i < n; i++)
            {
                long offset = offsets.Type == DataType.I64 ? BitConverter.ToInt64(raw, i * 8) : BitConverter.ToInt32(raw, i * 4);
                BitConverter.TryWriteBytes(bytes.AsSpan(i * 4, 4), checked((int)(i + offset)));
            }

            return bytes;
        });
    }
}
