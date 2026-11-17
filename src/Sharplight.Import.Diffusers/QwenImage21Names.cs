using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Maps Qwen-Image 2.1 transformer tensors (diffusers' <c>QwenImage21Transformer2DModel</c> names, and ComfyUI's / unsloth's GGUF files: the
/// same under <c>model.diffusion_model.</c> with the SwiGLU gate and up projections fused as <c>img_mlp.gate_up</c>, gate rows first) to the
/// canonical vocabulary of <c>QwenImage21TransformerGraphBuilder</c>. The zero-centred text norm's stored scale − 1 becomes the effective scale.
/// </summary>
internal static partial class QwenImage21Names
{
    private const string Canonical = "transformer.";

    private static readonly Dictionary<string, string> s_modules = new(StringComparer.Ordinal)
    {
        ["img_in"] = "img_in",
        ["txt_in.in_layer"] = "txt_in.in_layer",
        ["txt_in.out_layer"] = "txt_in.out_layer",
        ["time_text_embed.timestep_embedder.linear_1"] = "time_in.in_layer",
        ["time_text_embed.timestep_embedder.linear_2"] = "time_in.out_layer",
        ["modulation.1"] = "modulation",
        ["norm_out.linear"] = "final.mod",
        ["proj_out"] = "final.linear",
    };

    private static readonly Dictionary<string, string> s_blockParts = new(StringComparer.Ordinal)
    {
        ["attn.to_q"] = "attn.q",
        ["attn.to_k"] = "attn.k",
        ["attn.to_v"] = "attn.v",
        ["attn.to_out.0"] = "attn.o",
        ["attn.norm_q"] = "attn.norm_q",
        ["attn.norm_k"] = "attn.norm_k",
        ["img_mlp.gate_layer"] = "mlp.gate",
        ["img_mlp.proj"] = "mlp.up",
        ["img_mlp.out"] = "mlp.down",
    };

    /// <summary>Whether the names are a Qwen-Image 2.1 transformer (after removing <paramref name="prefix"/>).</summary>
    /// <param name="names">Tensor names.</param>
    /// <param name="prefix">Prefix in front of the transformer names (<c>""</c> or <c>model.diffusion_model.</c>).</param>
    public static bool IsTransformer(IReadOnlyCollection<string> names, string prefix)
        => names.Contains(prefix + "modulation.1.weight") && names.Contains(prefix + "txt_in.text_norm.weight");

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
            if (name == "txt_in.text_norm.weight")
            {
                weights.AddF32(Canonical + "txt_in.norm.weight", PlusOne(Canonical + "txt_in.norm.weight", weight));
                continue;
            }

            var fused = FusedMlp().Match(name);
            if (fused.Success)
            {
                long half = weight.Shape[0].Fixed / 2;
                string block = $"{Canonical}blocks.{fused.Groups[1].Value}.mlp.";
                weights.Add(block + "gate.weight", weight.GatherRows(block + "gate.weight", [(0, half)]));
                weights.Add(block + "up.weight", weight.GatherRows(block + "up.weight", [(half, half)]));
                continue;
            }

            weights.Add(Canonical + (CanonicalName(name) ?? throw new InvalidDataException($"Unexpected Qwen-Image 2.1 transformer tensor '{stored}'.")), weight);
        }
    }

    private static string? CanonicalName(string name)
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
            ? $"blocks.{block.Groups[1].Value}.{part}.{suffix}"
            : null;
    }

    /// <summary>The stored zero-centred scale plus one, as f32.</summary>
    internal static WeightInfo PlusOne(string name, WeightInfo stored) => WeightInfo.Derived(name, DataType.F32, new Shape(stored.Shape.Elements()), () =>
    {
        var probe = new CanonicalWeights();
        probe.AddF32(name, stored);
        byte[] bytes = probe.Weights[name].ReadAll();
        var values = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(bytes);
        for (int i = 0; i < values.Length; i++)
        {
            values[i] += 1f;
        }

        return bytes;
    });

    [GeneratedRegex(@"^transformer_blocks\.(\d+)\.(.+)$")]
    private static partial Regex Block();

    [GeneratedRegex(@"^transformer_blocks\.(\d+)\.img_mlp\.gate_up\.weight$")]
    private static partial Regex FusedMlp();
}
