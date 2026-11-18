using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Maps Qwen-Image transformer tensors (diffusers' <c>QwenImageTransformer2DModel</c> names, which the ComfyUI single files and city96's /
/// QuantStack's GGUF files keep) to the canonical <c>transformer.*</c> vocabulary of the FLUX transformer graph: dual-stream blocks
/// <c>double.{i}.{img,txt}_{mod,attn.q|k|v|proj|norm_q|norm_k,mlp.up|down}</c> (the projections stay separate: GGUF files quantize them
/// with different types), <c>img_in</c>, <c>txt_norm</c>, <c>txt_in</c>, <c>time_in.in_layer|out_layer</c>, <c>final.mod</c> (diffusers'
/// (scale, shift) halves swapped to FLUX's (shift, scale)) and <c>final.linear</c>.
/// </summary>
internal static partial class QwenImageNames
{
    private const string Canonical = "transformer.";

    /// <summary>
    /// The marker tensor of Qwen-Image-Edit-2511's GGUF files (an index of the zero timestep the reference images are modulated with,
    /// diffusers' <c>zero_cond_t</c>): it holds no weights and is not part of the canonical transformer.
    /// </summary>
    public const string ZeroConditionMarker = "__index_timestep_zero__";

    private static readonly Dictionary<string, string> s_blockParts = new(StringComparer.Ordinal)
    {
        ["img_mod.1"] = "img_mod",
        ["txt_mod.1"] = "txt_mod",
        ["attn.to_q"] = "img_attn.q",
        ["attn.to_k"] = "img_attn.k",
        ["attn.to_v"] = "img_attn.v",
        ["attn.to_out.0"] = "img_attn.proj",
        ["attn.norm_q"] = "img_attn.norm_q",
        ["attn.norm_k"] = "img_attn.norm_k",
        ["attn.add_q_proj"] = "txt_attn.q",
        ["attn.add_k_proj"] = "txt_attn.k",
        ["attn.add_v_proj"] = "txt_attn.v",
        ["attn.to_add_out"] = "txt_attn.proj",
        ["attn.norm_added_q"] = "txt_attn.norm_q",
        ["attn.norm_added_k"] = "txt_attn.norm_k",
        ["img_mlp.net.0.proj"] = "img_mlp.up",
        ["img_mlp.net.2"] = "img_mlp.down",
        ["txt_mlp.net.0.proj"] = "txt_mlp.up",
        ["txt_mlp.net.2"] = "txt_mlp.down",
    };

    private static readonly Dictionary<string, string> s_modules = new(StringComparer.Ordinal)
    {
        ["img_in"] = "img_in",
        ["txt_in"] = "txt_in",
        ["txt_norm"] = "txt_norm",
        ["time_text_embed.timestep_embedder.linear_1"] = "time_in.in_layer",
        ["time_text_embed.timestep_embedder.linear_2"] = "time_in.out_layer",
        ["norm_out.linear"] = "final.mod",
        ["proj_out"] = "final.linear",
    };

    /// <summary>Whether the names are a Qwen-Image transformer (after removing <paramref name="prefix"/>).</summary>
    /// <param name="names">Tensor names.</param>
    /// <param name="prefix">Prefix in front of the transformer names (<c>""</c> or <c>model.diffusion_model.</c>).</param>
    public static bool IsTransformer(IReadOnlyCollection<string> names, string prefix)
        => names.Contains(prefix + "transformer_blocks.0.img_mod.1.weight") && names.Contains(prefix + "txt_norm.weight");

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
            if (name == ZeroConditionMarker)
            {
                continue;
            }

            string canonical = Canonical + (CanonicalName(name) ?? throw new InvalidDataException($"Unexpected Qwen-Image transformer tensor '{stored}'."));
            weights.Add(canonical, canonical.StartsWith(Canonical + "final.mod.", StringComparison.Ordinal) ? SwapHalves(weight) : weight);
        }
    }

    internal static string? CanonicalName(string name)
    {
        int dot = name.LastIndexOf('.');
        if (dot < 0)
        {
            return null;
        }

        string module = name[..dot];
        string suffix = name[(dot + 1)..];
        if (s_modules.TryGetValue(module, out var mapped))
        {
            return $"{mapped}.{suffix}";
        }

        var block = Block().Match(module);
        return block.Success && s_blockParts.TryGetValue(block.Groups[2].Value, out var part)
            ? $"double.{block.Groups[1].Value}.{part}.{suffix}"
            : null;
    }

    /// <summary>The (shift, scale) order of the FLUX final modulation from diffusers' (scale, shift).</summary>
    private static WeightInfo SwapHalves(WeightInfo weight)
    {
        long half = weight.Shape[0].Fixed / 2;
        return weight.GatherRows(weight.Name, [(half, half), (0, half)]);
    }

    [GeneratedRegex(@"^transformer_blocks\.(\d+)\.(.+)$")]
    private static partial Regex Block();
}
