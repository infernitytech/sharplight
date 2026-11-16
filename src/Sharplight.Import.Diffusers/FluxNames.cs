using System.Globalization;
using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Maps FLUX transformer tensors to the canonical <c>transformer.*</c> vocabulary of the <c>flux</c> architecture (documented on
/// <c>FluxTransformerGraphBuilder</c>), from the original Black Forest Labs names (single files and the transformer-only GGUF files of
/// city96 / stable-diffusion.cpp: fused q/k/v, (shift, scale) final modulation; renamed zero-copy) and from diffusers'
/// <c>FluxTransformer2DModel</c> names (separate q/k/v and, in single-stream blocks, the MLP input are joined lazily; the final
/// modulation's (scale, shift) halves are swapped). Chroma's transformer (no modulation projections or embedders, the
/// <c>distilled_guidance_layer.*</c> approximator instead) maps through the same names.
/// </summary>
internal static partial class FluxNames
{
    private const string Canonical = "transformer.";

    /// <summary>Whether the names are an original-layout FLUX transformer (after removing <paramref name="prefix"/>).</summary>
    /// <param name="names">Tensor names.</param>
    /// <param name="prefix">Prefix in front of the transformer names (<c>""</c> or <c>model.diffusion_model.</c>).</param>
    public static bool IsOriginal(IEnumerable<string> names, string prefix)
        => names.Any(n => n == prefix + "double_blocks.0.img_attn.qkv.weight") && names.Any(n => n == prefix + "img_in.weight");

    /// <summary>Adds the original-layout transformer tensors (names after <paramref name="prefix"/>).</summary>
    /// <param name="tensors">The tensors by stored name.</param>
    /// <param name="prefix">Prefix in front of the transformer names.</param>
    /// <param name="weights">The canonical set.</param>
    /// <exception cref="InvalidDataException">A tensor has no canonical name.</exception>
    public static void AddOriginal(IReadOnlyDictionary<string, WeightInfo> tensors, string prefix, CanonicalWeights weights)
    {
        foreach (var (stored, weight) in tensors)
        {
            if (!stored.StartsWith(prefix, StringComparison.Ordinal))
            {
                continue;
            }

            string name = stored[prefix.Length..];
            string canonical = OriginalName(name) ?? throw new InvalidDataException($"Unexpected FLUX transformer tensor '{stored}'.");
            weights.Add(Canonical + canonical, weight);
        }
    }

    /// <summary>Adds diffusers <c>FluxTransformer2DModel</c> tensors (names without a component prefix).</summary>
    /// <param name="tensors">The tensors by diffusers name.</param>
    /// <param name="weights">The canonical set.</param>
    /// <exception cref="InvalidDataException">A tensor has no canonical name.</exception>
    public static void AddDiffusers(IReadOnlyDictionary<string, WeightInfo> tensors, CanonicalWeights weights)
    {
        var fused = new Dictionary<string, SortedDictionary<int, WeightInfo>>(StringComparer.Ordinal);
        foreach (var (name, weight) in tensors)
        {
            var (canonical, part) = DiffusersName(name);
            if (canonical is null)
            {
                throw new InvalidDataException($"Unexpected FLUX transformer tensor '{name}'.");
            }

            if (part < 0)
            {
                weights.Add(Canonical + canonical, canonical.StartsWith("final.mod.", StringComparison.Ordinal) ? SwapHalves(weight) : weight);
                continue;
            }

            if (!fused.TryGetValue(canonical, out var parts))
            {
                fused[canonical] = parts = [];
            }

            parts[part] = weight;
        }

        foreach (var (canonical, parts) in fused)
        {
            weights.Add(Canonical + canonical, WeightInfo.ConcatRows(Canonical + canonical, [.. parts.Values]));
        }
    }

    /// <summary>The (shift, scale) order of the original final modulation from diffusers' (scale, shift).</summary>
    private static WeightInfo SwapHalves(WeightInfo weight)
    {
        long half = weight.Shape[0].Fixed / 2;
        return weight.GatherRows(weight.Name, [(half, half), (0, half)]);
    }

    private static string? OriginalName(string name)
    {
        if (ApproximatorName(name) is { } approximator)
        {
            return approximator;
        }

        var m = OriginalPart().Match(name);
        if (!m.Success)
        {
            return null;
        }

        string module = m.Groups[1].Value;
        string suffix = m.Groups[2].Value;
        string? mapped = module switch
        {
            "img_in" or "txt_in" or "time_in.in_layer" or "time_in.out_layer" or "vector_in.in_layer" or "vector_in.out_layer"
                or "guidance_in.in_layer" or "guidance_in.out_layer" => module,
            "final_layer.adaLN_modulation.1" => "final.mod",
            "final_layer.linear" => "final.linear",
            _ => null,
        };
        if (mapped is not null)
        {
            return $"{mapped}.{suffix}";
        }

        var block = OriginalBlock().Match(module);
        if (!block.Success)
        {
            return null;
        }

        string index = block.Groups[2].Value;
        string part = block.Groups[3].Value;
        if (block.Groups[1].Value == "double_blocks")
        {
            string? inner = part switch
            {
                "img_mod.lin" => "img_mod",
                "txt_mod.lin" => "txt_mod",
                "img_attn.qkv" or "txt_attn.qkv" or "img_attn.proj" or "txt_attn.proj" => part,
                "img_attn.norm.query_norm" => "img_attn.norm_q",
                "img_attn.norm.key_norm" => "img_attn.norm_k",
                "txt_attn.norm.query_norm" => "txt_attn.norm_q",
                "txt_attn.norm.key_norm" => "txt_attn.norm_k",
                "img_mlp.0" => "img_mlp.up",
                "img_mlp.2" => "img_mlp.down",
                "txt_mlp.0" => "txt_mlp.up",
                "txt_mlp.2" => "txt_mlp.down",
                _ => null,
            };
            return inner is null ? null : $"double.{index}.{inner}.{(suffix == "scale" ? "weight" : suffix)}";
        }

        string? single = part switch
        {
            "modulation.lin" => "mod",
            "linear1" or "linear2" => part,
            "norm.query_norm" => "norm_q",
            "norm.key_norm" => "norm_k",
            _ => null,
        };
        return single is null ? null : $"single.{index}.{single}.{(suffix == "scale" ? "weight" : suffix)}";
    }

    /// <summary>The canonical name of a diffusers tensor, and its row block index when several are joined into it (else −1).</summary>
    private static (string? Canonical, int Part) DiffusersName(string name)
    {
        if (ApproximatorName(name) is { } approximator)
        {
            return (approximator, -1);
        }

        int dot = name.LastIndexOf('.');
        string module = name[..dot];
        string suffix = name[(dot + 1)..];
        string? mapped = module switch
        {
            "x_embedder" => "img_in",
            "context_embedder" => "txt_in",
            "time_text_embed.timestep_embedder.linear_1" => "time_in.in_layer",
            "time_text_embed.timestep_embedder.linear_2" => "time_in.out_layer",
            "time_text_embed.text_embedder.linear_1" => "vector_in.in_layer",
            "time_text_embed.text_embedder.linear_2" => "vector_in.out_layer",
            "time_text_embed.guidance_embedder.linear_1" => "guidance_in.in_layer",
            "time_text_embed.guidance_embedder.linear_2" => "guidance_in.out_layer",
            "norm_out.linear" => "final.mod",
            "proj_out" => "final.linear",
            _ => null,
        };
        if (mapped is not null)
        {
            return ($"{mapped}.{suffix}", -1);
        }

        var block = DiffusersBlock().Match(module);
        if (!block.Success)
        {
            return (null, -1);
        }

        string index = block.Groups[2].Value;
        string part = block.Groups[3].Value;
        if (block.Groups[1].Value == "transformer_blocks")
        {
            string p = $"double.{index}.";
            return part switch
            {
                "norm1.linear" => (p + "img_mod." + suffix, -1),
                "norm1_context.linear" => (p + "txt_mod." + suffix, -1),
                "attn.to_q" => (p + "img_attn.qkv." + suffix, 0),
                "attn.to_k" => (p + "img_attn.qkv." + suffix, 1),
                "attn.to_v" => (p + "img_attn.qkv." + suffix, 2),
                "attn.add_q_proj" => (p + "txt_attn.qkv." + suffix, 0),
                "attn.add_k_proj" => (p + "txt_attn.qkv." + suffix, 1),
                "attn.add_v_proj" => (p + "txt_attn.qkv." + suffix, 2),
                "attn.norm_q" => (p + "img_attn.norm_q." + suffix, -1),
                "attn.norm_k" => (p + "img_attn.norm_k." + suffix, -1),
                "attn.norm_added_q" => (p + "txt_attn.norm_q." + suffix, -1),
                "attn.norm_added_k" => (p + "txt_attn.norm_k." + suffix, -1),
                "attn.to_out.0" => (p + "img_attn.proj." + suffix, -1),
                "attn.to_add_out" => (p + "txt_attn.proj." + suffix, -1),
                "ff.net.0.proj" => (p + "img_mlp.up." + suffix, -1),
                "ff.net.2" => (p + "img_mlp.down." + suffix, -1),
                "ff_context.net.0.proj" => (p + "txt_mlp.up." + suffix, -1),
                "ff_context.net.2" => (p + "txt_mlp.down." + suffix, -1),
                _ => (null, -1),
            };
        }

        string s = $"single.{index}.";
        return part switch
        {
            "norm.linear" => (s + "mod." + suffix, -1),
            "attn.to_q" => (s + "linear1." + suffix, 0),
            "attn.to_k" => (s + "linear1." + suffix, 1),
            "attn.to_v" => (s + "linear1." + suffix, 2),
            "proj_mlp" => (s + "linear1." + suffix, 3),
            "attn.norm_q" => (s + "norm_q." + suffix, -1),
            "attn.norm_k" => (s + "norm_k." + suffix, -1),
            "proj_out" => (s + "linear2." + suffix, -1),
            _ => (null, -1),
        };
    }

    /// <summary>
    /// The canonical name of a tensor of Chroma's modulation approximator (<c>distilled_guidance_layer.*</c>: the same names in the original
    /// and diffusers layouts, except the MLP layers (<c>in_layer</c> / <c>out_layer</c>, diffusers <c>linear_1</c> / <c>linear_2</c>) and the
    /// norm scales (<c>scale</c>, diffusers <c>weight</c>)), or null.
    /// </summary>
    private static string? ApproximatorName(string name)
    {
        const string Stored = "distilled_guidance_layer.";
        if (!name.StartsWith(Stored, StringComparison.Ordinal))
        {
            return null;
        }

        string rest = name[Stored.Length..].Replace(".linear_1.", ".in_layer.", StringComparison.Ordinal).Replace(".linear_2.", ".out_layer.", StringComparison.Ordinal);
        if (rest.StartsWith("norms.", StringComparison.Ordinal) && rest.EndsWith(".scale", StringComparison.Ordinal))
        {
            rest = rest[..^".scale".Length] + ".weight";
        }

        return "approximator." + rest;
    }

    /// <summary>The number of approximator layers in canonical weights (0 for FLUX).</summary>
    /// <param name="weights">The canonical weights.</param>
    public static int CountApproximatorLayers(IReadOnlyDictionary<string, WeightInfo> weights)
    {
        int count = 0;
        while (weights.ContainsKey(string.Create(CultureInfo.InvariantCulture, $"{Canonical}approximator.norms.{count}.weight")))
        {
            count++;
        }

        return count;
    }

    /// <summary>The number of blocks of a kind in canonical weights (<c>double</c> or <c>single</c>).</summary>
    /// <param name="weights">The canonical weights.</param>
    /// <param name="kind">The block kind.</param>
    public static int CountBlocks(IReadOnlyDictionary<string, WeightInfo> weights, string kind)
    {
        int count = 0;
        while (weights.ContainsKey(string.Create(CultureInfo.InvariantCulture, $"{Canonical}{kind}.{count}.{(kind == "double" ? "img_attn.qkv" : "linear1")}.weight")))
        {
            count++;
        }

        return count;
    }

    [GeneratedRegex(@"^(.+)\.(weight|bias|scale)$")]
    private static partial Regex OriginalPart();

    [GeneratedRegex(@"^(double_blocks|single_blocks)\.(\d+)\.(.+)$")]
    private static partial Regex OriginalBlock();

    [GeneratedRegex(@"^(transformer_blocks|single_transformer_blocks)\.(\d+)\.(.+)$")]
    private static partial Regex DiffusersBlock();
}
