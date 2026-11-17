using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Maps Krea 2 transformer tensors to the canonical vocabulary of <c>Krea2TransformerGraphBuilder</c>: the original Krea names (the release's
/// <c>turbo.safetensors</c> / <c>raw.safetensors</c>, ComfyUI's files, the GGUF files; optionally under <c>model.diffusion_model.</c>) and
/// diffusers' <c>Krea2Transformer2DModel</c> names. Every RMS norm of the model has a zero-centred scale: the stored scale + 1 becomes the
/// effective one (f32). The blocks' modulation tables, stored flattened in the original files, become [6, D].
/// </summary>
internal static partial class Krea2Names
{
    private const string Canonical = "transformer.";

    // Original module -> canonical module (outside the blocks).
    private static readonly Dictionary<string, string> s_original = new(StringComparer.Ordinal)
    {
        ["first"] = "img_in",
        ["tmlp.0"] = "time_in.in_layer",
        ["tmlp.2"] = "time_in.out_layer",
        ["tproj.1"] = "time_mod",
        ["txtmlp.1"] = "txt_in.in_layer",
        ["txtmlp.3"] = "txt_in.out_layer",
        ["last.linear"] = "final.linear",
        ["txtfusion.projector"] = "text_fusion.projector",
    };

    // diffusers module -> canonical module (outside the blocks).
    private static readonly Dictionary<string, string> s_diffusers = new(StringComparer.Ordinal)
    {
        ["img_in"] = "img_in",
        ["time_embed.linear_1"] = "time_in.in_layer",
        ["time_embed.linear_2"] = "time_in.out_layer",
        ["time_mod_proj"] = "time_mod",
        ["txt_in.linear_1"] = "txt_in.in_layer",
        ["txt_in.linear_2"] = "txt_in.out_layer",
        ["final_layer.linear"] = "final.linear",
        ["text_fusion.projector"] = "text_fusion.projector",
    };

    // Block part (original, diffusers) -> canonical part.
    private static readonly Dictionary<string, string> s_parts = new(StringComparer.Ordinal)
    {
        ["attn.wq"] = "attn.q",
        ["attn.wk"] = "attn.k",
        ["attn.wv"] = "attn.v",
        ["attn.wo"] = "attn.o",
        ["attn.gate"] = "attn.gate",
        ["mlp.gate"] = "mlp.gate",
        ["mlp.up"] = "mlp.up",
        ["mlp.down"] = "mlp.down",
        ["attn.to_q"] = "attn.q",
        ["attn.to_k"] = "attn.k",
        ["attn.to_v"] = "attn.v",
        ["attn.to_out.0"] = "attn.o",
        ["attn.to_gate"] = "attn.gate",
        ["ff.gate"] = "mlp.gate",
        ["ff.up"] = "mlp.up",
        ["ff.down"] = "mlp.down",
    };

    // Zero-centred norm scales (full stored name after the block prefix) -> canonical norm.
    private static readonly Dictionary<string, string> s_norms = new(StringComparer.Ordinal)
    {
        ["attn.qknorm.qnorm.scale"] = "attn.norm_q.weight",
        ["attn.qknorm.knorm.scale"] = "attn.norm_k.weight",
        ["prenorm.scale"] = "norm1.weight",
        ["postnorm.scale"] = "norm2.weight",
        ["attn.norm_q.weight"] = "attn.norm_q.weight",
        ["attn.norm_k.weight"] = "attn.norm_k.weight",
        ["norm1.weight"] = "norm1.weight",
        ["norm2.weight"] = "norm2.weight",
    };

    /// <summary>
    /// Tensors some GGUF conversions carry that are not part of the model (vantagewithai's files embed a 6144 × 6144 picture as
    /// <c>last.up</c> / <c>last.down</c>, described by <c>egg_*</c> metadata); the reference implementations never read them.
    /// </summary>
    private static readonly HashSet<string> s_ignored = new(StringComparer.Ordinal) { "last.up.weight", "last.down.weight" };

    /// <summary>The prefix of the transformer names when <paramref name="names"/> are a Krea 2 transformer in the original names, else null.</summary>
    /// <param name="names">Tensor names.</param>
    public static string? OriginalPrefix(IReadOnlyCollection<string> names)
        => new[] { "", "model.diffusion_model." }.FirstOrDefault(prefix => names.Contains(prefix + "txtfusion.projector.weight") && names.Contains(prefix + "tproj.1.weight"));

    /// <summary>Whether <paramref name="names"/> are diffusers' Krea 2 transformer.</summary>
    /// <param name="names">Tensor names (without the component prefix).</param>
    public static bool IsDiffusers(IReadOnlyCollection<string> names) => names.Contains("text_fusion.projector.weight") && names.Contains("time_mod_proj.weight");

    /// <summary>Adds the transformer tensors (names after <paramref name="prefix"/>; others are skipped).</summary>
    /// <param name="tensors">The tensors by stored name.</param>
    /// <param name="prefix">Prefix in front of the transformer names.</param>
    /// <param name="weights">The canonical set.</param>
    /// <exception cref="InvalidDataException">A transformer tensor has no canonical name.</exception>
    public static void Add(IReadOnlyDictionary<string, WeightInfo> tensors, string prefix, CanonicalWeights weights)
    {
        foreach (var (stored, weight) in tensors)
        {
            if (!stored.StartsWith(prefix, StringComparison.Ordinal))
            {
                continue;
            }

            string name = stored[prefix.Length..];
            if (s_ignored.Contains(name))
            {
                continue;
            }

            AddOne(name, stored, weight, weights);
        }
    }

    private static void AddOne(string name, string stored, WeightInfo weight, CanonicalWeights weights)
    {
        switch (name)
        {
            case "txtmlp.0.scale" or "txt_in.norm.weight":
                AddNorm(Canonical + "txt_in.norm.weight", weight, weights);
                return;
            case "last.norm.scale" or "final_layer.norm.weight":
                AddNorm(Canonical + "final.norm.weight", weight, weights);
                return;
            case "txtfusion.projector.weight" or "text_fusion.projector.weight":
                // [1, layers]; GGUF readers may drop the leading unit dimension.
                weights.Add(Canonical + "text_fusion.projector.weight", weight.As(Canonical + "text_fusion.projector.weight", new Shape(1, weight.Shape.Elements())));
                return;
            case "last.modulation.lin" or "final_layer.scale_shift_table":
                weights.AddF32(Canonical + "final.mod", weight.As(Canonical + "final.mod", new Shape(2, weight.Shape.Elements() / 2)));
                return;
        }

        var block = Block().Match(name);
        if (block.Success)
        {
            string canonicalBlock = block.Groups[1].Value switch
            {
                "blocks" or "transformer_blocks" => "blocks",
                "txtfusion.layerwise_blocks" or "text_fusion.layerwise_blocks" => "text_fusion.layerwise",
                _ => "text_fusion.refiner",
            };
            string p = $"{Canonical}{canonicalBlock}.{block.Groups[2].Value}.";
            string rest = block.Groups[3].Value;
            if (s_norms.TryGetValue(rest, out var norm))
            {
                AddNorm(p + norm, weight, weights);
                return;
            }

            if (rest is "mod.lin" or "scale_shift_table")
            {
                weights.AddF32(p + "mod", weight.As(p + "mod", new Shape(6, weight.Shape.Elements() / 6)));
                return;
            }

            int dot = rest.LastIndexOf('.');
            if (dot > 0 && s_parts.TryGetValue(rest[..dot], out var part))
            {
                weights.Add(p + part + rest[dot..], weight);
                return;
            }
        }
        else
        {
            int dot = name.LastIndexOf('.');
            if (dot > 0 && (s_original.TryGetValue(name[..dot], out var module) || s_diffusers.TryGetValue(name[..dot], out module)))
            {
                weights.Add(Canonical + module + name[dot..], weight);
                return;
            }
        }

        throw new InvalidDataException($"Unexpected Krea 2 transformer tensor '{stored}'.");
    }

    private static void AddNorm(string name, WeightInfo weight, CanonicalWeights weights) => weights.AddF32(name, QwenImage21Names.PlusOne(name, weight));

    [GeneratedRegex(@"^(blocks|transformer_blocks|txtfusion\.layerwise_blocks|txtfusion\.refiner_blocks|text_fusion\.layerwise_blocks|text_fusion\.refiner_blocks)\.(\d+)\.(.+)$")]
    private static partial Regex Block();
}
