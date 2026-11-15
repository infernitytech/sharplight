using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Maps diffusers UNet2DConditionModel and AutoencoderKL tensor names to the canonical latent-diffusion names (see the
/// <c>LatentDiffusionAdapter</c> remarks in Sharplight.Architectures for the vocabulary) and adds them to a
/// <see cref="CanonicalWeights"/> set.
/// </summary>
internal static partial class DiffusersNames
{
    /// <summary>Adds one UNet tensor under its canonical <c>unet.*</c> name.</summary>
    /// <param name="name">Diffusers name (without a component prefix).</param>
    /// <param name="weight">The stored tensor.</param>
    /// <param name="weights">The canonical set.</param>
    /// <exception cref="InvalidDataException">The name is not part of a supported UNet.</exception>
    public static void AddUNet(string name, WeightInfo weight, CanonicalWeights weights)
    {
        string s = name;
        s = Replace(s, "time_embedding.linear_1.", "time_embedding.linear1.");
        s = Replace(s, "time_embedding.linear_2.", "time_embedding.linear2.");
        s = Replace(s, "add_embedding.linear_1.", "class_embedding.linear1.");
        s = Replace(s, "add_embedding.linear_2.", "class_embedding.linear2.");
        s = Replace(s, "conv_in.", "input_conv.");
        s = Replace(s, "conv_norm_out.", "output_norm.");
        s = Replace(s, "conv_out.", "output_conv.");
        s = DownBlock().Replace(s, "down.$1.$2.$3.");
        s = UpBlock().Replace(s, "up.$1.$2.$3.");
        s = Sampler().Replace(s, "$1.$2.${3}sample.");
        s = MidBlock().Replace(s, "mid.$1.$2.");
        s = s.Replace("mid.attn.0.", "mid.attn.", StringComparison.Ordinal);
        s = ResnetParts(s);
        if (s == name)
        {
            throw new InvalidDataException($"Unexpected UNet tensor '{name}'.");
        }

        AddTransformerOrPlain("unet." + s, weight, weights);
    }

    /// <summary>Adds one VAE tensor under its canonical <c>vae.*</c> name.</summary>
    /// <param name="name">Diffusers name (without a component prefix).</param>
    /// <param name="weight">The stored tensor.</param>
    /// <param name="weights">The canonical set.</param>
    public static void AddVae(string name, WeightInfo weight, CanonicalWeights weights)
    {
        string s = name;
        s = Replace(s, "coder.conv_in.", "coder.input_conv.");
        s = Replace(s, "coder.conv_norm_out.", "coder.output_norm.");
        s = Replace(s, "coder.conv_out.", "coder.output_conv.");
        s = VaeBlock().Replace(s, "$1.$2.$3.$4.");
        s = Sampler().Replace(s, "$1.$2.${3}sample.");
        s = VaeMid().Replace(s, "$1.mid.$2.$3.");
        s = ResnetParts(s);
        var attention = VaeAttention().Match(s);
        if (attention.Success)
        {
            string part = attention.Groups[2].Value switch
            {
                "group_norm" => "norm",
                "to_q" or "query" => "q",
                "to_k" or "key" => "k",
                "to_v" or "value" => "v",
                _ => "o",
            };
            string canonical = $"vae.{attention.Groups[1].Value}.mid.attn.{part}.{attention.Groups[3].Value}";
            if (part == "norm")
            {
                weights.Add(canonical, weight);
            }
            else
            {
                weights.AddLinear(canonical, weight);
            }

            return;
        }

        weights.Add("vae." + s, weight);
    }

    /// <summary>Spatial-transformer tensors get their own sub-names; everything else keeps the translated name.</summary>
    private static void AddTransformerOrPlain(string canonical, WeightInfo weight, CanonicalWeights weights)
    {
        var block = TransformerBlock().Match(canonical);
        if (block.Success)
        {
            string prefix = $"{block.Groups[1].Value}.blocks.{block.Groups[2].Value}.";
            string rest = block.Groups[3].Value;
            if (rest.StartsWith("ff.net.0.proj.", StringComparison.Ordinal))
            {
                string suffix = rest["ff.net.0.proj.".Length..];
                weights.AddHalves($"{prefix}ff.up.{suffix}", $"{prefix}ff.gate.{suffix}", weight);
                return;
            }

            rest = Replace(rest, "ff.net.2.", "ff.down.");
            rest = AttentionPart().Replace(rest, m => $"{m.Groups[1].Value}.{AttentionProjection(m.Groups[2].Value)}.");
            weights.Add(prefix + rest, weight);
            return;
        }

        if (TransformerProjection().IsMatch(canonical))
        {
            weights.AddLinear(canonical, weight);
            return;
        }

        weights.Add(canonical, weight);
    }

    private static string AttentionProjection(string part) => part switch
    {
        "to_q" => "q",
        "to_k" => "k",
        "to_v" => "v",
        _ => "o",
    };

    private static string ResnetParts(string s) => ResnetPart().Replace(s, m => m.Groups[1].Value switch
    {
        "time_emb_proj" => "time_projection",
        "conv_shortcut" => "shortcut",
        var other => other,
    });

    private static string Replace(string s, string from, string to)
        => s.StartsWith(from, StringComparison.Ordinal) || s.Contains("." + from, StringComparison.Ordinal) ? s.Replace(from, to, StringComparison.Ordinal) : s;

    [GeneratedRegex(@"^down_blocks\.(\d+)\.(resnets|attentions)\.(\d+)\.")]
    private static partial Regex DownBlock();

    [GeneratedRegex(@"^up_blocks\.(\d+)\.(resnets|attentions)\.(\d+)\.")]
    private static partial Regex UpBlock();

    [GeneratedRegex(@"^((?:encoder\.|decoder\.)?)(?:down_blocks|up_blocks)\.(\d+)\.(down|up)samplers\.0\.conv\.")]
    private static partial Regex SamplerPrefix();

    [GeneratedRegex(@"(down|up)_blocks\.(\d+)\.(down|up)samplers\.0\.conv\.")]
    private static partial Regex Sampler();

    [GeneratedRegex(@"^mid_block\.(resnets|attentions)\.(\d+)\.")]
    private static partial Regex MidBlock();

    [GeneratedRegex(@"^(encoder|decoder)\.(down|up)_blocks\.(\d+)\.resnets\.(\d+)\.")]
    private static partial Regex VaeBlock();

    [GeneratedRegex(@"^(encoder|decoder)\.mid_block\.(resnets|attentions)\.(\d+)\.")]
    private static partial Regex VaeMid();

    [GeneratedRegex(@"^(encoder|decoder)\.mid\.attentions\.0\.(group_norm|to_q|to_k|to_v|to_out\.0|query|key|value|proj_attn)\.(weight|bias)$")]
    private static partial Regex VaeAttention();

    [GeneratedRegex(@"(?<=\.res\.\d+\.)(time_emb_proj|conv_shortcut)(?=\.)")]
    private static partial Regex ResnetPart();

    [GeneratedRegex(@"^(.*\.attn(?:\.\d+)?)\.transformer_blocks\.(\d+)\.(.*)$")]
    private static partial Regex TransformerBlock();

    [GeneratedRegex(@"\.attn(?:\.\d+)?\.proj_(in|out)\.weight$")]
    private static partial Regex TransformerProjection();

    [GeneratedRegex(@"^(attn[12])\.(to_q|to_k|to_v|to_out\.0)\.")]
    private static partial Regex AttentionPart();
}
