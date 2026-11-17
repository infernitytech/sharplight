using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Maps HunyuanVideo transformer tensors to the canonical names of the <c>hunyuan_video</c> architecture: the original HunyuanVideo names
/// (as GGUF and ComfyUI files store them) under <c>transformer.</c>; diffusers' split projections (q, k, v; the single blocks' q, k, v and
/// MLP input) are concatenated and its final modulation's (scale, shift) halves swapped to (shift, scale).
/// </summary>
internal static partial class HunyuanVideoNames
{
    private const string Canonical = "transformer.";

    private static readonly (string Diffusers, string Original)[] s_plain =
    [
        ("time_text_embed.timestep_embedder.linear_1.", "time_in.mlp.0."),
        ("time_text_embed.timestep_embedder.linear_2.", "time_in.mlp.2."),
        ("time_text_embed.guidance_embedder.linear_1.", "guidance_in.mlp.0."),
        ("time_text_embed.guidance_embedder.linear_2.", "guidance_in.mlp.2."),
        ("time_text_embed.text_embedder.linear_1.", "vector_in.in_layer."),
        ("time_text_embed.text_embedder.linear_2.", "vector_in.out_layer."),
        ("context_embedder.time_text_embed.timestep_embedder.linear_1.", "txt_in.t_embedder.mlp.0."),
        ("context_embedder.time_text_embed.timestep_embedder.linear_2.", "txt_in.t_embedder.mlp.2."),
        ("context_embedder.time_text_embed.text_embedder.linear_1.", "txt_in.c_embedder.linear_1."),
        ("context_embedder.time_text_embed.text_embedder.linear_2.", "txt_in.c_embedder.linear_2."),
        ("context_embedder.proj_in.", "txt_in.input_embedder."),
        ("x_embedder.proj.", "img_in.proj."),
        ("proj_out.", "final_layer.linear."),
        ("norm_out.linear.", "final_layer.adaLN_modulation.1."),
    ];

    private static readonly Dictionary<string, (string Original, int Part)> s_refiner = new(StringComparer.Ordinal)
    {
        ["norm1"] = ("norm1", -1),
        ["norm2"] = ("norm2", -1),
        ["attn.to_q"] = ("self_attn_qkv", 0),
        ["attn.to_k"] = ("self_attn_qkv", 1),
        ["attn.to_v"] = ("self_attn_qkv", 2),
        ["attn.to_out.0"] = ("self_attn_proj", -1),
        ["ff.net.0.proj"] = ("mlp.fc1", -1),
        ["ff.net.2"] = ("mlp.fc2", -1),
        ["norm_out.linear"] = ("adaLN_modulation.1", -1),
    };

    private static readonly Dictionary<string, (string Original, int Part)> s_double = new(StringComparer.Ordinal)
    {
        ["norm1.linear"] = ("img_mod.linear", -1),
        ["norm1_context.linear"] = ("txt_mod.linear", -1),
        ["attn.to_q"] = ("img_attn_qkv", 0),
        ["attn.to_k"] = ("img_attn_qkv", 1),
        ["attn.to_v"] = ("img_attn_qkv", 2),
        ["attn.add_q_proj"] = ("txt_attn_qkv", 0),
        ["attn.add_k_proj"] = ("txt_attn_qkv", 1),
        ["attn.add_v_proj"] = ("txt_attn_qkv", 2),
        ["attn.norm_q"] = ("img_attn_q_norm", -1),
        ["attn.norm_k"] = ("img_attn_k_norm", -1),
        ["attn.norm_added_q"] = ("txt_attn_q_norm", -1),
        ["attn.norm_added_k"] = ("txt_attn_k_norm", -1),
        ["attn.to_out.0"] = ("img_attn_proj", -1),
        ["attn.to_add_out"] = ("txt_attn_proj", -1),
        ["ff.net.0.proj"] = ("img_mlp.fc1", -1),
        ["ff.net.2"] = ("img_mlp.fc2", -1),
        ["ff_context.net.0.proj"] = ("txt_mlp.fc1", -1),
        ["ff_context.net.2"] = ("txt_mlp.fc2", -1),
    };

    private static readonly Dictionary<string, (string Original, int Part)> s_single = new(StringComparer.Ordinal)
    {
        ["norm.linear"] = ("modulation.linear", -1),
        ["attn.to_q"] = ("linear1", 0),
        ["attn.to_k"] = ("linear1", 1),
        ["attn.to_v"] = ("linear1", 2),
        ["proj_mlp"] = ("linear1", 3),
        ["attn.norm_q"] = ("q_norm", -1),
        ["attn.norm_k"] = ("k_norm", -1),
        ["proj_out"] = ("linear2", -1),
    };

    /// <summary>Whether the names are the original HunyuanVideo transformer's (GGUF / ComfyUI), under <paramref name="prefix"/>.</summary>
    /// <param name="names">Tensor names.</param>
    /// <param name="prefix">The prefix before the transformer names (empty, or <c>model.diffusion_model.</c>).</param>
    public static bool IsOriginal(IReadOnlyCollection<string> names, string prefix)
        => names.Contains(prefix + "double_blocks.0.img_attn_qkv.weight") && names.Contains(prefix + "txt_in.input_embedder.weight");

    /// <summary>Adds original-name tensors (names after <paramref name="prefix"/>).</summary>
    /// <param name="tensors">The tensors by stored name.</param>
    /// <param name="prefix">The prefix before the transformer names.</param>
    /// <param name="weights">The canonical set.</param>
    public static void AddOriginal(IReadOnlyDictionary<string, WeightInfo> tensors, string prefix, CanonicalWeights weights)
    {
        foreach (var (name, weight) in tensors)
        {
            if (name.StartsWith(prefix, StringComparison.Ordinal))
            {
                Add(Canonical + name[prefix.Length..], weight, weights);
            }
        }
    }

    /// <summary>Adds diffusers <c>HunyuanVideoTransformer3DModel</c> tensors (names without the component prefix).</summary>
    /// <param name="tensors">The tensors by diffusers name.</param>
    /// <param name="weights">The canonical set.</param>
    /// <exception cref="InvalidDataException">A tensor has no canonical name.</exception>
    public static void AddDiffusers(IReadOnlyDictionary<string, WeightInfo> tensors, CanonicalWeights weights)
    {
        var fused = new Dictionary<string, SortedDictionary<int, WeightInfo>>(StringComparer.Ordinal);
        foreach (var (name, weight) in tensors)
        {
            var (original, part) = DiffusersName(name);
            if (original is null)
            {
                throw new InvalidDataException($"Unexpected HunyuanVideo transformer tensor '{name}'.");
            }

            if (part < 0)
            {
                Add(Canonical + original, original.StartsWith("final_layer.adaLN_modulation.", StringComparison.Ordinal) ? SwapHalves(weight) : weight, weights);
                continue;
            }

            if (!fused.TryGetValue(original, out var parts))
            {
                fused[original] = parts = [];
            }

            parts[part] = weight;
        }

        foreach (var (original, parts) in fused)
        {
            Add(Canonical + original, WeightInfo.ConcatRows(Canonical + original, [.. parts.Values]), weights);
        }
    }

    private static void Add(string canonical, WeightInfo weight, CanonicalWeights weights)
    {
        if (canonical == Canonical + "img_in.proj.weight")
        {
            long outputs = weight.Shape[0].Fixed;
            weights.Add(canonical, weight.As(canonical, new Shape(outputs, weight.Shape.Elements() / outputs)));
            return;
        }

        weights.Add(canonical, weight.As(canonical));
    }

    /// <summary>The (shift, scale) order of the original final modulation from diffusers' (scale, shift).</summary>
    private static WeightInfo SwapHalves(WeightInfo weight)
    {
        long half = weight.Shape[0].Fixed / 2;
        return weight.GatherRows(weight.Name, [(half, half), (0, half)]);
    }

    private static (string? Original, int Part) DiffusersName(string name)
    {
        int dot = name.LastIndexOf('.');
        string module = name[..dot];
        string kind = name[dot..];
        var match = Block().Match(module);
        if (match.Success)
        {
            var table = match.Groups[1].Value switch
            {
                "context_embedder.token_refiner.refiner_blocks" => s_refiner,
                "transformer_blocks" => s_double,
                _ => s_single,
            };
            string block = match.Groups[1].Value switch
            {
                "context_embedder.token_refiner.refiner_blocks" => "txt_in.individual_token_refiner.blocks",
                "transformer_blocks" => "double_blocks",
                _ => "single_blocks",
            };
            return table.TryGetValue(match.Groups[3].Value, out var mapped)
                ? ($"{block}.{match.Groups[2].Value}.{mapped.Original}{kind}", mapped.Part)
                : (null, -1);
        }

        foreach (var (diffusers, original) in s_plain)
        {
            if ((module + ".").Equals(diffusers, StringComparison.Ordinal))
            {
                return (original + kind[1..], -1);
            }
        }

        return (null, -1);
    }

    [GeneratedRegex(@"^(context_embedder\.token_refiner\.refiner_blocks|transformer_blocks|single_transformer_blocks)\.(\d+)\.(.+)$")]
    private static partial Regex Block();
}
