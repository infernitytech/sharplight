using System.Globalization;
using System.Text.RegularExpressions;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Turns kohya-ss LoRA module names (the module's dotted name with every <c>.</c> replaced by <c>_</c>) back into dotted names: the
/// underscores inside the known multi-word names (<c>down_blocks</c>, <c>to_q</c>, <c>img_attn</c>, ...) stay, the others become dots.
/// SDXL LoRAs name the UNet's SGM modules (<c>input_blocks.4.1</c>), which are mapped to diffusers' modules as diffusers'
/// <c>_maybe_map_sgm_blocks_to_diffusers</c> does.
/// </summary>
internal static partial class KohyaNames
{
    private static readonly HashSet<string> s_words = new(StringComparer.Ordinal)
    {
        // UNet (diffusers and SGM)
        "down_blocks", "up_blocks", "mid_block", "transformer_blocks", "to_q", "to_k", "to_v", "to_out", "proj_in", "proj_out", "time_emb_proj",
        "conv_shortcut", "conv_in", "conv_out", "time_embedding", "linear_1", "linear_2", "add_embedding", "input_blocks", "output_blocks",
        "middle_block", "in_layers", "out_layers", "emb_layers", "skip_connection", "label_emb", "time_embed", "conv_norm_out",
        // CLIP text encoders
        "text_model", "self_attn", "q_proj", "k_proj", "v_proj", "out_proj", "text_projection", "layer_norm1", "layer_norm2",
        // FLUX (original and diffusers names)
        "double_blocks", "single_blocks", "img_attn", "txt_attn", "img_mlp", "txt_mlp", "img_mod", "txt_mod", "img_in", "txt_in", "time_in",
        "vector_in", "guidance_in", "in_layer", "out_layer", "final_layer", "adaLN_modulation", "single_transformer_blocks", "add_q_proj",
        "add_k_proj", "add_v_proj", "to_add_out", "ff_context", "proj_mlp", "norm_out", "time_text_embed", "timestep_embedder",
        "text_embedder", "guidance_embedder", "x_embedder", "context_embedder",
    };

    /// <summary>The dotted module name of a kohya name.</summary>
    /// <param name="name">The kohya module name without its <c>lora_*_</c> prefix.</param>
    public static string Dotted(string name)
    {
        var parts = name.Split('_');
        var result = new List<string>();
        for (int i = 0; i < parts.Length; i++)
        {
            // The longest known word starting here (up to four parts: single_transformer_blocks, time_text_embed, ...).
            int take = 1;
            for (int n = Math.Min(4, parts.Length - i); n > 1; n--)
            {
                if (s_words.Contains(string.Join('_', parts, i, n)))
                {
                    take = n;
                    break;
                }
            }

            result.Add(string.Join('_', parts, i, take));
            i += take - 1;
        }

        return string.Join('.', result);
    }

    /// <summary>
    /// The diffusers UNet module of a kohya UNet module name: SD 1.x LoRAs already name diffusers modules; SDXL LoRAs name the SGM
    /// UNet's (<c>input_blocks</c>, <c>middle_block</c>, <c>output_blocks</c> with <c>in_layers</c> / <c>out_layers</c> / ...).
    /// </summary>
    /// <param name="name">The kohya module name without <c>lora_unet_</c>.</param>
    /// <param name="layersPerBlock">The UNet's residual blocks per down level (SDXL: 2).</param>
    public static string UNet(string name, int layersPerBlock)
    {
        string dotted = Dotted(name);
        return dotted.StartsWith("input_blocks.", StringComparison.Ordinal) || dotted.StartsWith("middle_block.", StringComparison.Ordinal)
            || dotted.StartsWith("output_blocks.", StringComparison.Ordinal) || dotted.StartsWith("time_embed.", StringComparison.Ordinal)
            || dotted.StartsWith("label_emb.", StringComparison.Ordinal) || dotted.StartsWith("out.", StringComparison.Ordinal)
            ? Sgm(dotted, layersPerBlock)
            : dotted;
    }

    private static string Sgm(string dotted, int layersPerBlock)
    {
        var top = new (string From, string To)[]
        {
            ("input_blocks.0.0", "conv_in"), ("time_embed.0", "time_embedding.linear_1"), ("time_embed.2", "time_embedding.linear_2"),
            ("label_emb.0.0", "add_embedding.linear_1"), ("label_emb.0.2", "add_embedding.linear_2"), ("out.0", "conv_norm_out"), ("out.2", "conv_out"),
        };
        foreach (var (from, to) in top)
        {
            if (dotted == from || dotted.StartsWith(from + ".", StringComparison.Ordinal))
            {
                return to + dotted[from.Length..];
            }
        }

        var middle = SgmMiddle().Match(dotted);
        if (middle.Success)
        {
            string part = middle.Groups[1].Value switch
            {
                "0" => "mid_block.resnets.0",
                "1" => "mid_block.attentions.0",
                _ => "mid_block.resnets.1",
            };
            return part + ResidualParts(middle.Groups[2].Value);
        }

        var m = SgmBlock().Match(dotted);
        if (!m.Success)
        {
            return dotted;
        }

        string kind = m.Groups[1].Value;
        int block = int.Parse(m.Groups[2].Value, CultureInfo.InvariantCulture);
        int inner = int.Parse(m.Groups[3].Value, CultureInfo.InvariantCulture);
        string tail = m.Groups[4].Value;
        int perLevel = layersPerBlock + 1;
        string module;
        switch (kind)
        {
            case "input_blocks":
            {
                int level = (block - 1) / perLevel;
                int layer = (block - 1) % perLevel;
                module = tail.StartsWith(".op", StringComparison.Ordinal)
                    ? $"down_blocks.{level}.downsamplers.0"
                    : $"down_blocks.{level}.{(inner == 0 ? "resnets" : "attentions")}.{layer}";
                tail = tail.Replace(".op", ".conv", StringComparison.Ordinal);
                break;
            }

            default:
            {
                int level = block / perLevel;
                int layer = block % perLevel;
                bool upsampler = inner == 2 || (inner == 1 && tail.StartsWith(".conv", StringComparison.Ordinal));
                module = upsampler ? $"up_blocks.{level}.upsamplers.0" : $"up_blocks.{level}.{(inner == 0 ? "resnets" : "attentions")}.{layer}";
                break;
            }
        }

        return module + ResidualParts(tail);
    }

    /// <summary>SGM residual-block parts → diffusers names.</summary>
    private static string ResidualParts(string tail)
        => tail.Replace(".in_layers.0", ".norm1", StringComparison.Ordinal)
            .Replace(".in_layers.2", ".conv1", StringComparison.Ordinal)
            .Replace(".out_layers.0", ".norm2", StringComparison.Ordinal)
            .Replace(".out_layers.3", ".conv2", StringComparison.Ordinal)
            .Replace(".emb_layers.1", ".time_emb_proj", StringComparison.Ordinal)
            .Replace(".skip_connection", ".conv_shortcut", StringComparison.Ordinal);

    [GeneratedRegex(@"^(input_blocks|output_blocks)\.(\d+)\.(\d+)(.*)$")]
    private static partial Regex SgmBlock();

    [GeneratedRegex(@"^middle_block\.(\d+)(.*)$")]
    private static partial Regex SgmMiddle();
}
