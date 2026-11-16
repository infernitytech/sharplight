using System.Globalization;
using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Maps SD3 / SD3.5 transformer tensors to the canonical <c>transformer.*</c> vocabulary of the <c>sd3</c> architecture (documented on
/// <c>Sd3TransformerGraphBuilder</c>), from the original MMDiT names (Stability's single files and the GGUF files of city96 /
/// stable-diffusion.cpp: fused q/k/v, (shift, scale) for the final and the last context modulation; renamed zero-copy) and from diffusers'
/// <c>SD3Transformer2DModel</c> names (separate q/k/v joined lazily; the (scale, shift) halves of <c>norm_out</c> and of the last block's
/// <c>norm1_context</c> swapped). The patch-embedding convolution becomes a linear weight [D, C·p²] and the position grid [G², D].
/// </summary>
internal static partial class Sd3Names
{
    private const string Canonical = "transformer.";

    /// <summary>Whether the names are an original-layout SD3 transformer (after removing <paramref name="prefix"/>).</summary>
    /// <param name="names">Tensor names.</param>
    /// <param name="prefix">Prefix in front of the transformer names (<c>""</c> or <c>model.diffusion_model.</c>).</param>
    public static bool IsOriginal(IEnumerable<string> names, string prefix)
        => names.Any(n => n == prefix + "joint_blocks.0.x_block.attn.qkv.weight") && names.Any(n => n == prefix + "x_embedder.proj.weight");

    /// <summary>Adds the original-layout transformer tensors (names after <paramref name="prefix"/>; other tensors are ignored).</summary>
    /// <param name="tensors">The tensors by stored name.</param>
    /// <param name="prefix">Prefix in front of the transformer names.</param>
    /// <param name="weights">The canonical set.</param>
    /// <exception cref="InvalidDataException">A transformer tensor has no canonical name.</exception>
    public static void AddOriginal(IReadOnlyDictionary<string, WeightInfo> tensors, string prefix, CanonicalWeights weights)
    {
        foreach (var (stored, weight) in tensors)
        {
            if (!stored.StartsWith(prefix, StringComparison.Ordinal) || (prefix.Length == 0 && IsOtherComponent(stored)))
            {
                continue;
            }

            string name = stored[prefix.Length..];
            string canonical = OriginalName(name) ?? throw new InvalidDataException($"Unexpected SD3 transformer tensor '{stored}'.");
            weights.Add(Canonical + canonical, Reshaped(Canonical + canonical, weight));
        }
    }

    /// <summary>Adds diffusers <c>SD3Transformer2DModel</c> tensors (names without a component prefix).</summary>
    /// <param name="tensors">The tensors by diffusers name.</param>
    /// <param name="weights">The canonical set.</param>
    /// <exception cref="InvalidDataException">A tensor has no canonical name.</exception>
    public static void AddDiffusers(IReadOnlyDictionary<string, WeightInfo> tensors, CanonicalWeights weights)
    {
        int lastBlock = tensors.Keys.Select(BlockIndex).DefaultIfEmpty(-1).Max();
        var fused = new Dictionary<string, SortedDictionary<int, WeightInfo>>(StringComparer.Ordinal);
        foreach (var (name, weight) in tensors)
        {
            var (canonical, part) = DiffusersName(name);
            if (canonical is null)
            {
                throw new InvalidDataException($"Unexpected SD3 transformer tensor '{name}'.");
            }

            if (part < 0)
            {
                bool swap = canonical.StartsWith("final.mod.", StringComparison.Ordinal)
                    || canonical.StartsWith(string.Create(CultureInfo.InvariantCulture, $"joint.{lastBlock}.txt_mod."), StringComparison.Ordinal);
                weights.Add(Canonical + canonical, Reshaped(Canonical + canonical, swap ? SwapHalves(weight) : weight));
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

    /// <summary>The number of joint blocks in canonical weights.</summary>
    /// <param name="weights">The canonical weights.</param>
    public static int CountBlocks(IReadOnlyDictionary<string, WeightInfo> weights)
    {
        int count = 0;
        while (weights.ContainsKey(string.Create(CultureInfo.InvariantCulture, $"{Canonical}joint.{count}.img_mod.weight")))
        {
            count++;
        }

        return count;
    }

    /// <summary>Tensors of a combined single file that are not the transformer (VAE, text encoders).</summary>
    private static bool IsOtherComponent(string name)
        => name.StartsWith("first_stage_model.", StringComparison.Ordinal) || name.StartsWith("text_encoders.", StringComparison.Ordinal);

    /// <summary>The patch embedding as a linear weight [D, C·p²] and the position grid as [G², D].</summary>
    private static WeightInfo Reshaped(string canonical, WeightInfo weight)
    {
        if (canonical == Canonical + "patch_embed.weight" && weight.Shape.Rank == 4)
        {
            long d = weight.Shape[0].Fixed;
            return weight.As(weight.Name, new Tensors.Shape(d, weight.Shape.Elements() / d));
        }

        if (canonical == Canonical + "pos_embed" && weight.Shape.Rank == 3)
        {
            return weight.As(weight.Name, new Tensors.Shape(weight.Shape[1].Fixed, weight.Shape[2].Fixed));
        }

        return weight;
    }

    /// <summary>The (shift, scale) order of the original modulation from diffusers' (scale, shift).</summary>
    private static WeightInfo SwapHalves(WeightInfo weight)
    {
        long half = weight.Shape[0].Fixed / 2;
        return weight.GatherRows(weight.Name, [(half, half), (0, half)]);
    }

    private static int BlockIndex(string name)
    {
        var m = DiffusersBlock().Match(name);
        return m.Success ? int.Parse(m.Groups[1].Value, CultureInfo.InvariantCulture) : -1;
    }

    private static string? OriginalName(string name)
    {
        if (name == "pos_embed")
        {
            return "pos_embed";
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
            "x_embedder.proj" => "patch_embed",
            "t_embedder.mlp.0" => "time_in.in_layer",
            "t_embedder.mlp.2" => "time_in.out_layer",
            "y_embedder.mlp.0" => "vector_in.in_layer",
            "y_embedder.mlp.2" => "vector_in.out_layer",
            "context_embedder" => "txt_in",
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

        string index = block.Groups[1].Value;
        string stream = block.Groups[2].Value == "x_block" ? "img" : "txt";
        string? inner = block.Groups[3].Value switch
        {
            "adaLN_modulation.1" => "mod",
            "attn.qkv" => "attn.qkv",
            "attn.proj" => "attn.proj",
            "attn.ln_q" => "attn.norm_q",
            "attn.ln_k" => "attn.norm_k",
            "attn2.qkv" => "attn2.qkv",
            "attn2.proj" => "attn2.proj",
            "attn2.ln_q" => "attn2.norm_q",
            "attn2.ln_k" => "attn2.norm_k",
            "mlp.fc1" => "mlp.up",
            "mlp.fc2" => "mlp.down",
            _ => null,
        };
        return inner is null ? null : $"joint.{index}.{stream}_{inner}.{suffix}";
    }

    /// <summary>The canonical name of a diffusers tensor, and its row block index when several are joined into it (else −1).</summary>
    private static (string? Canonical, int Part) DiffusersName(string name)
    {
        if (name == "pos_embed.pos_embed")
        {
            return ("pos_embed", -1);
        }

        int dot = name.LastIndexOf('.');
        string module = name[..dot];
        string suffix = name[(dot + 1)..];
        string? mapped = module switch
        {
            "pos_embed.proj" => "patch_embed",
            "context_embedder" => "txt_in",
            "time_text_embed.timestep_embedder.linear_1" => "time_in.in_layer",
            "time_text_embed.timestep_embedder.linear_2" => "time_in.out_layer",
            "time_text_embed.text_embedder.linear_1" => "vector_in.in_layer",
            "time_text_embed.text_embedder.linear_2" => "vector_in.out_layer",
            "norm_out.linear" => "final.mod",
            "proj_out" => "final.linear",
            _ => null,
        };
        if (mapped is not null)
        {
            return ($"{mapped}.{suffix}", -1);
        }

        var block = DiffusersBlockPart().Match(module);
        if (!block.Success)
        {
            return (null, -1);
        }

        string p = $"joint.{block.Groups[1].Value}.";
        return block.Groups[2].Value switch
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
            "attn2.to_q" => (p + "img_attn2.qkv." + suffix, 0),
            "attn2.to_k" => (p + "img_attn2.qkv." + suffix, 1),
            "attn2.to_v" => (p + "img_attn2.qkv." + suffix, 2),
            "attn2.norm_q" => (p + "img_attn2.norm_q." + suffix, -1),
            "attn2.norm_k" => (p + "img_attn2.norm_k." + suffix, -1),
            "attn2.to_out.0" => (p + "img_attn2.proj." + suffix, -1),
            "ff.net.0.proj" => (p + "img_mlp.up." + suffix, -1),
            "ff.net.2" => (p + "img_mlp.down." + suffix, -1),
            "ff_context.net.0.proj" => (p + "txt_mlp.up." + suffix, -1),
            "ff_context.net.2" => (p + "txt_mlp.down." + suffix, -1),
            _ => (null, -1),
        };
    }

    [GeneratedRegex(@"^(.+)\.(weight|bias)$")]
    private static partial Regex OriginalPart();

    [GeneratedRegex(@"^joint_blocks\.(\d+)\.(x_block|context_block)\.(.+)$")]
    private static partial Regex OriginalBlock();

    [GeneratedRegex(@"^transformer_blocks\.(\d+)\.")]
    private static partial Regex DiffusersBlock();

    [GeneratedRegex(@"^transformer_blocks\.(\d+)\.(.+)$")]
    private static partial Regex DiffusersBlockPart();
}
