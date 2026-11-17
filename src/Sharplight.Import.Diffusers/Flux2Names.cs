using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Maps FLUX.2 transformer tensors to the canonical FLUX transformer vocabulary (documented on <c>FluxTransformerGraphBuilder</c>), from the
/// original Black Forest Labs names (the single file in the model repository, ComfyUI's, the GGUF files of unsloth / leejet: FLUX.1's names
/// plus the shared <c>double_stream_modulation_img|txt.lin</c> and <c>single_stream_modulation.lin</c>) and from diffusers'
/// <c>Flux2Transformer2DModel</c> names (separate q / k / v joined lazily, <c>ff.linear_in|out</c>, <c>to_qkv_mlp_proj</c>, the final
/// modulation's (scale, shift) halves swapped). The shared modulations become <c>transformer.mod_img</c>, <c>mod_txt</c>, <c>mod_single</c>.
/// </summary>
internal static partial class Flux2Names
{
    private const string Canonical = "transformer.";

    private static readonly Dictionary<string, string> s_originalShared = new(StringComparer.Ordinal)
    {
        ["double_stream_modulation_img.lin"] = "mod_img",
        ["double_stream_modulation_txt.lin"] = "mod_txt",
        ["single_stream_modulation.lin"] = "mod_single",
    };

    private static readonly Dictionary<string, string> s_diffusersModules = new(StringComparer.Ordinal)
    {
        ["x_embedder"] = "img_in",
        ["context_embedder"] = "txt_in",
        ["time_guidance_embed.timestep_embedder.linear_1"] = "time_in.in_layer",
        ["time_guidance_embed.timestep_embedder.linear_2"] = "time_in.out_layer",
        ["time_guidance_embed.guidance_embedder.linear_1"] = "guidance_in.in_layer",
        ["time_guidance_embed.guidance_embedder.linear_2"] = "guidance_in.out_layer",
        ["double_stream_modulation_img.linear"] = "mod_img",
        ["double_stream_modulation_txt.linear"] = "mod_txt",
        ["single_stream_modulation.linear"] = "mod_single",
        ["norm_out.linear"] = "final.mod",
        ["proj_out"] = "final.linear",
    };

    /// <summary>The prefix of the transformer names when the names are a FLUX.2 transformer in the original layout, else null.</summary>
    /// <param name="names">The tensor names.</param>
    public static string? OriginalPrefix(IReadOnlyCollection<string> names)
        => new[] { "", "model.diffusion_model." }.FirstOrDefault(p => names.Contains(p + "double_stream_modulation_img.lin.weight"));

    /// <summary>Whether the names are a diffusers <c>Flux2Transformer2DModel</c>.</summary>
    /// <param name="names">The tensor names.</param>
    public static bool IsDiffusers(IReadOnlyCollection<string> names) => names.Contains("double_stream_modulation_img.linear.weight");

    /// <summary>Adds the original-layout tensors (names after <paramref name="prefix"/>).</summary>
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
            int dot = name.LastIndexOf('.');
            string? canonical = s_originalShared.TryGetValue(name[..dot], out var shared) ? $"{shared}.{name[(dot + 1)..]}" : FluxNames.OriginalName(name);
            weights.Add(Canonical + (canonical ?? throw new InvalidDataException($"Unexpected FLUX.2 transformer tensor '{stored}'.")), weight);
        }
    }

    /// <summary>Adds diffusers <c>Flux2Transformer2DModel</c> tensors (names without a component prefix).</summary>
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
                throw new InvalidDataException($"Unexpected FLUX.2 transformer tensor '{name}'.");
            }

            if (part < 0)
            {
                bool swap = canonical.StartsWith("final.mod.", StringComparison.Ordinal);
                long half = weight.Shape[0].Fixed / 2;
                weights.Add(Canonical + canonical, swap ? weight.GatherRows(weight.Name, [(half, half), (0, half)]) : weight);
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

    private static (string? Canonical, int Part) DiffusersName(string name)
    {
        int dot = name.LastIndexOf('.');
        string module = name[..dot];
        string suffix = name[(dot + 1)..];
        if (s_diffusersModules.TryGetValue(module, out var mapped))
        {
            return ($"{mapped}.{suffix}", -1);
        }

        var block = Block().Match(module);
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
                "ff.linear_in" => (p + "img_mlp.up." + suffix, -1),
                "ff.linear_out" => (p + "img_mlp.down." + suffix, -1),
                "ff_context.linear_in" => (p + "txt_mlp.up." + suffix, -1),
                "ff_context.linear_out" => (p + "txt_mlp.down." + suffix, -1),
                _ => (null, -1),
            };
        }

        string s = $"single.{index}.";
        return part switch
        {
            "attn.to_qkv_mlp_proj" => (s + "linear1." + suffix, -1),
            "attn.to_out" => (s + "linear2." + suffix, -1),
            "attn.norm_q" => (s + "norm_q." + suffix, -1),
            "attn.norm_k" => (s + "norm_k." + suffix, -1),
            _ => (null, -1),
        };
    }

    [GeneratedRegex(@"^(transformer_blocks|single_transformer_blocks)\.(\d+)\.(.+)$")]
    private static partial Regex Block();
}
