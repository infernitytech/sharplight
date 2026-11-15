using System.Globalization;
using System.Text.RegularExpressions;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Translates the tensor names of original (CompVis / Stability "LDM") checkpoints into diffusers names, which
/// <see cref="DiffusersNames"/> then maps to the canonical ones. Single-file safetensors checkpoints and stable-diffusion.cpp
/// GGUF files use these names: <c>model.diffusion_model.input_blocks.1.0.in_layers.0.weight</c>,
/// <c>first_stage_model.decoder.up.3.block.0.conv1.weight</c>, ... The block numbering is resolved from the tensors present,
/// so any number of levels and residual blocks per level is accepted.
/// </summary>
internal static partial class LdmNames
{
    /// <summary>Maps the UNet tensor names (without the <c>model.diffusion_model.</c> prefix) to diffusers UNet names.</summary>
    /// <param name="names">All UNet tensor names.</param>
    /// <returns>LDM name → diffusers name for every tensor that belongs to the UNet.</returns>
    public static Dictionary<string, string> UNet(IReadOnlyCollection<string> names)
    {
        var blocks = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["time_embed.0"] = "time_embedding.linear_1",
            ["time_embed.2"] = "time_embedding.linear_2",
            ["label_emb.0.0"] = "add_embedding.linear_1",
            ["label_emb.0.2"] = "add_embedding.linear_2",
            ["input_blocks.0.0"] = "conv_in",
            ["out.0"] = "conv_norm_out",
            ["out.2"] = "conv_out",
            ["middle_block.0"] = "mid_block.resnets.0",
            ["middle_block.1"] = "mid_block.attentions.0",
            ["middle_block.2"] = "mid_block.resnets.1",
        };

        int inputCount = BlockCount(names, "input_blocks.");
        int level = 0;
        int index = 0;
        int resnetsPerLevel = 0;
        for (int n = 1; n < inputCount; n++)
        {
            if (names.Contains($"input_blocks.{n}.0.op.weight"))
            {
                blocks[$"input_blocks.{n}.0.op"] = $"down_blocks.{level}.downsamplers.0.conv";
                level++;
                index = 0;
                continue;
            }

            blocks[$"input_blocks.{n}.0"] = $"down_blocks.{level}.resnets.{index}";
            blocks[$"input_blocks.{n}.1"] = $"down_blocks.{level}.attentions.{index}";
            index++;
            if (level == 0)
            {
                resnetsPerLevel = index;
            }
        }

        int outputCount = BlockCount(names, "output_blocks.");
        int perUpLevel = resnetsPerLevel + 1;
        for (int n = 0; n < outputCount; n++)
        {
            int upLevel = n / perUpLevel;
            int j = n % perUpLevel;
            blocks[$"output_blocks.{n}.0"] = $"up_blocks.{upLevel}.resnets.{j}";
            if (names.Contains($"output_blocks.{n}.1.conv.weight"))
            {
                blocks[$"output_blocks.{n}.1.conv"] = $"up_blocks.{upLevel}.upsamplers.0.conv";
            }
            else
            {
                blocks[$"output_blocks.{n}.1"] = $"up_blocks.{upLevel}.attentions.{j}";
                blocks[$"output_blocks.{n}.2.conv"] = $"up_blocks.{upLevel}.upsamplers.0.conv";
            }
        }

        var map = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (string name in names)
        {
            if (Translate(name, blocks) is { } translated)
            {
                map[name] = ResnetPart().Replace(translated, m => ResnetParts[m.Groups[1].Value]);
            }
        }

        return map;
    }

    /// <summary>Maps the VAE tensor names (without the <c>first_stage_model.</c> prefix) to diffusers AutoencoderKL names.</summary>
    /// <param name="names">All VAE tensor names.</param>
    /// <returns>LDM name → diffusers name for every tensor that belongs to the VAE.</returns>
    public static Dictionary<string, string> Vae(IReadOnlyCollection<string> names)
    {
        int decoderLevels = names.Select(n => VaeUpLevel().Match(n)).Where(m => m.Success)
            .Select(m => int.Parse(m.Groups[1].Value, CultureInfo.InvariantCulture)).DefaultIfEmpty(-1).Max() + 1;
        var map = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (string name in names)
        {
            string s = name;
            s = VaeDownBlock().Replace(s, "encoder.down_blocks.$1.resnets.$2.");
            s = VaeDownSample().Replace(s, "encoder.down_blocks.$1.downsamplers.0.conv.");
            s = VaeUpBlock().Replace(s, m => $"decoder.up_blocks.{decoderLevels - 1 - int.Parse(m.Groups[1].Value, CultureInfo.InvariantCulture)}.resnets.{m.Groups[2].Value}.");
            s = VaeUpSample().Replace(s, m => $"decoder.up_blocks.{decoderLevels - 1 - int.Parse(m.Groups[1].Value, CultureInfo.InvariantCulture)}.upsamplers.0.conv.");
            s = s.Replace(".mid.block_1.", ".mid_block.resnets.0.", StringComparison.Ordinal)
                .Replace(".mid.block_2.", ".mid_block.resnets.1.", StringComparison.Ordinal)
                .Replace(".mid.attn_1.norm.", ".mid_block.attentions.0.group_norm.", StringComparison.Ordinal)
                .Replace(".mid.attn_1.q.", ".mid_block.attentions.0.to_q.", StringComparison.Ordinal)
                .Replace(".mid.attn_1.k.", ".mid_block.attentions.0.to_k.", StringComparison.Ordinal)
                .Replace(".mid.attn_1.v.", ".mid_block.attentions.0.to_v.", StringComparison.Ordinal)
                .Replace(".mid.attn_1.proj_out.", ".mid_block.attentions.0.to_out.0.", StringComparison.Ordinal)
                .Replace(".nin_shortcut.", ".conv_shortcut.", StringComparison.Ordinal)
                .Replace("coder.norm_out.", "coder.conv_norm_out.", StringComparison.Ordinal);
            map[name] = s;
        }

        return map;
    }

    /// <summary>Number of numbered blocks under a prefix such as <c>input_blocks.</c> (one more than the highest index).</summary>
    private static int BlockCount(IEnumerable<string> names, string prefix)
    {
        int count = 0;
        foreach (string name in names)
        {
            if (!name.StartsWith(prefix, StringComparison.Ordinal))
            {
                continue;
            }

            int end = name.IndexOf('.', prefix.Length);
            if (end > prefix.Length && int.TryParse(name.AsSpan(prefix.Length, end - prefix.Length), NumberStyles.None, CultureInfo.InvariantCulture, out int n))
            {
                count = Math.Max(count, n + 1);
            }
        }

        return count;
    }

    /// <summary>Replaces the longest block prefix of <paramref name="name"/> found in <paramref name="blocks"/>; null when none matches.</summary>
    private static string? Translate(string name, Dictionary<string, string> blocks)
    {
        for (int end = name.LastIndexOf('.'); end > 0; end = name.LastIndexOf('.', end - 1))
        {
            if (blocks.TryGetValue(name[..end], out var replacement))
            {
                return replacement + name[end..];
            }
        }

        return null;
    }

    private static readonly Dictionary<string, string> ResnetParts = new(StringComparer.Ordinal)
    {
        ["in_layers.0"] = "norm1",
        ["in_layers.2"] = "conv1",
        ["emb_layers.1"] = "time_emb_proj",
        ["out_layers.0"] = "norm2",
        ["out_layers.3"] = "conv2",
        ["skip_connection"] = "conv_shortcut",
    };

    [GeneratedRegex(@"(?<=resnets\.\d+\.)(in_layers\.0|in_layers\.2|emb_layers\.1|out_layers\.0|out_layers\.3|skip_connection)(?=\.)")]
    private static partial Regex ResnetPart();

    [GeneratedRegex(@"^decoder\.up\.(\d+)\.")]
    private static partial Regex VaeUpLevel();

    [GeneratedRegex(@"^encoder\.down\.(\d+)\.block\.(\d+)\.")]
    private static partial Regex VaeDownBlock();

    [GeneratedRegex(@"^encoder\.down\.(\d+)\.downsample\.conv\.")]
    private static partial Regex VaeDownSample();

    [GeneratedRegex(@"^decoder\.up\.(\d+)\.block\.(\d+)\.")]
    private static partial Regex VaeUpBlock();

    [GeneratedRegex(@"^decoder\.up\.(\d+)\.upsample\.conv\.")]
    private static partial Regex VaeUpSample();
}
