using System.Text.Json;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Raon-1 mini (HF <c>raon1_mini</c>, KRAFTON's remote code): Raon delta attention (RDA) layers, NoPE gated latent attention layers (the 64 "rope"
/// channels are never rotated: <c>rope.style none</c>), a dense SiTU-GLU first layer and sigmoid-routed SiTU MoE layers with a shared expert, all on
/// a 4-stream Raon_mHC residual (Helmert-chart stream mixing, streams averaged before the output norm). Layer kinds come from the 1-based
/// <c>rda_config.rda_layers</c>; the RDA write gate <c>rda_w_proj</c> becomes <c>linear_attention.write.weight</c>; the ten parameters of each mHC site
/// (<c>{self_attn,mlp}_raon_mhc.*</c>) are packed into <c>{attention,mlp}_hc.{fn.weight,base,scale}</c> as llama.cpp's converter packs them; the MoE
/// below <c>moe.</c> maps like DeepSeek's <c>mlp.</c>; the reasoning-effort steering vectors become <c>mlp.steering</c> with the default effort's
/// factor (<c>reasoning_effort_steering</c>). MXFP8 routed experts are resolved before the mapping (<see cref="HfFp8Weights"/>).
/// </summary>
internal sealed partial class LatentDecoderHfMapping
{
    private static bool IsRaon1(HfConfig c) => c.ModelType == "raon1_mini";

    /// <summary>The ten parameters of one Raon_mHC site, in llama.cpp's packing order (fn rows, base entries, scale entries).</summary>
    private static readonly string[] s_raonMhcParts =
        ["phi_pre", "phi_post", "res.phi_z", "b_pre", "b_post", "res.b_z", "alpha_pre", "alpha_post", "res.alpha_z", "res.eta"];

    private static void AddRaon1HyperParameters(Dictionary<string, object> md, string a, HfConfig c)
    {
        md[a + "attention.latent_norm_eps"] = (float)(c.GetDouble("mla_latent_norm_eps") ?? 1e-6);
        md[a + "rope.style"] = "none";
        md[a + "mlp.activation"] = "situ";
        md[a + "mlp.situ_beta"] = (float)(c.GetDouble("activation_situ_beta") ?? 4.0);
        md[a + "mlp.situ_linear_beta"] = (float)(c.GetDouble("activation_situ_linear_beta") ?? 25.0);

        // The decay gate: lower_bound·sigmoid(...) with the config's bound (−5 when absent); an explicit null selects −softplus.
        if (LinearConfig(c) is { } rda)
        {
            if (!rda.TryGetProperty("gate_lower_bound", out var bound))
            {
                md[a + "linear_attention.decay_bound"] = -5f;
            }
            else if (bound.ValueKind == JsonValueKind.Number)
            {
                md[a + "linear_attention.decay_bound"] = (float)bound.GetDouble();
            }
        }

        md[a + "hyper_connections.streams"] = c.GetInt("raon_mhc_streams") ?? 4;
        md[a + "hyper_connections.norm_eps"] = (float)(c.GetDouble("raon_mhc_selector_eps") ?? 1e-5);
        md[a + "hyper_connections.magnitude"] = (float)(c.GetDouble("raon_mhc_post_factor") ?? 2.0);
        md[a + "hyper_connections.eps"] = 0f;
        md[a + "hyper_connections.pre_eps"] = 0f;
        md[a + "hyper_connections.chart_mixing"] = true;
        md[a + "hyper_connections.mixing_transposed"] = true;

        if (RaonDefaultEffortScale(c) is { } scale and not 0f)
        {
            md[a + "mlp.steering_scale"] = scale;
        }
    }

    /// <summary>
    /// The factor of the default reasoning effort (<c>reasoning_effort_steering.alpha[default]</c>), which the reference applies to every token of a
    /// request without a reasoning effort; null without steering.
    /// </summary>
    private static float? RaonDefaultEffortScale(HfConfig c)
    {
        if (c.Get("reasoning_effort_steering") is not { ValueKind: JsonValueKind.Object } steering
            || !steering.TryGetProperty("alpha", out var alpha) || alpha.ValueKind != JsonValueKind.Object)
        {
            return null;
        }

        string? effort = steering.TryGetProperty("default", out var d) && d.ValueKind == JsonValueKind.String ? d.GetString() : null;
        return effort is not null && alpha.TryGetProperty(effort, out var value) && value.ValueKind == JsonValueKind.Number ? (float)value.GetDouble() : null;
    }

    /// <summary>Renames the MoE block (<c>moe.</c>) to the DeepSeek-style <c>mlp.</c> suffixes <see cref="TryMapLayer"/> knows.</summary>
    private static string NormalizeRaon1Suffix(string suffix)
        => suffix.StartsWith("moe.", StringComparison.Ordinal) ? "mlp." + suffix["moe.".Length..] : suffix;

    /// <summary>
    /// Collects a Raon_mHC parameter (<c>self_attn_raon_mhc.*</c> → <c>attention_hc</c>, <c>mlp_raon_mhc.*</c> → <c>mlp_hc</c>) or maps the steering
    /// vector; false for other tensors.
    /// </summary>
    private static bool TryMapRaon1Layer(string prefix, string suffix, WeightInfo w, HfWeightCollector collector,
        Dictionary<string, Dictionary<string, WeightInfo>> mhc)
    {
        if (suffix == "mlp.reasoning_effort_steering")
        {
            collector.Add(HfTransforms.ToF32(w, prefix + "mlp.steering", new Shape(w.Shape.Elements())));
            return true;
        }

        if (suffix == "self_attn.rda_w_proj.weight")
        {
            collector.Add(HfTransforms.Plain(w, prefix + "linear_attention.write.weight"));
            return true;
        }

        foreach (var (site, module) in (ReadOnlySpan<(string, string)>)[("self_attn_raon_mhc.", "attention_hc."), ("mlp_raon_mhc.", "mlp_hc.")])
        {
            if (suffix.StartsWith(site, StringComparison.Ordinal) && s_raonMhcParts.Contains(suffix[site.Length..]))
            {
                string key = prefix + module;
                if (!mhc.TryGetValue(key, out var parts))
                {
                    mhc[key] = parts = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
                }

                parts[suffix[site.Length..]] = w;
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Every mHC site packed: <c>fn.weight</c> [2N + (N−1)², N·H] = [phi_pre; phi_post; phi_z] (stored dtype), <c>base</c> f32 [2N + (N−1)²] =
    /// [b_pre; b_post; b_z] and <c>scale</c> f32 [4] = [alpha_pre, alpha_post, alpha_z, eta].
    /// </summary>
    /// <exception cref="InvalidDataException">A site lacks one of its parameters.</exception>
    private static void AddRaonMhcModules(Dictionary<string, Dictionary<string, WeightInfo>> mhc, HfWeightCollector collector)
    {
        foreach (var (module, parts) in mhc.OrderBy(p => p.Key, StringComparer.Ordinal))
        {
            if (s_raonMhcParts.FirstOrDefault(part => !parts.ContainsKey(part)) is { } missing)
            {
                throw new InvalidDataException($"{module}: the Raon_mHC site lacks {missing}.");
            }

            var fn = new[] { parts["phi_pre"], parts["phi_post"], parts["res.phi_z"] };
            long width = fn[0].Shape[-1].Fixed;
            long rows = fn.Sum(p => p.Shape[0].Fixed);
            collector.Add(WeightInfo.Derived(module + "fn.weight", DataType.F32, new Shape(rows, width),
                () => HfTransforms.EncodeF32([.. fn.SelectMany(p => HfTransforms.DecodeF32(p))])));
            var bases = new[] { parts["b_pre"], parts["b_post"], parts["res.b_z"] };
            collector.Add(WeightInfo.Derived(module + "base", DataType.F32, new Shape(bases.Sum(p => p.Shape.Elements())),
                () => HfTransforms.EncodeF32([.. bases.SelectMany(p => HfTransforms.DecodeF32(p))])));
            var scales = new[] { parts["alpha_pre"], parts["alpha_post"], parts["res.alpha_z"], parts["res.eta"] };
            collector.Add(WeightInfo.Derived(module + "scale", DataType.F32, new Shape(scales.Length),
                () => HfTransforms.EncodeF32([.. scales.SelectMany(p => HfTransforms.DecodeF32(p))])));
        }
    }
}
