using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Maps diffusers <c>UNet2DConditionModel</c> and <c>AutoencoderKL</c> tensor names to the canonical latent-diffusion names (the
/// vocabulary is documented on <c>LatentDiffusionAdapter</c> in Sharplight.Architectures) and adds them to a
/// <see cref="CanonicalWeights"/> set. Original (LDM) checkpoints reach this class through <see cref="LdmNames"/>.
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
        string s = Prefix(name, "time_embedding.linear_1.", "time_embedding.linear1.");
        s = Prefix(s, "time_embedding.linear_2.", "time_embedding.linear2.");
        s = Prefix(s, "add_embedding.linear_1.", "class_embedding.linear1.");
        s = Prefix(s, "add_embedding.linear_2.", "class_embedding.linear2.");
        s = Prefix(s, "conv_in.", "input_conv.");
        s = Prefix(s, "conv_norm_out.", "output_norm.");
        s = Prefix(s, "conv_out.", "output_conv.");
        s = LevelPart().Replace(s, m => $"{m.Groups[1].Value}.{m.Groups[2].Value}.{(m.Groups[3].Value == "resnets" ? "res" : "attn")}.{m.Groups[4].Value}.");
        s = Downsampler().Replace(s, "down.$1.downsample.");
        s = Upsampler().Replace(s, "up.$1.upsample.");
        s = MidPart().Replace(s, m => $"mid.{(m.Groups[1].Value == "resnets" ? "res" : "attn")}.{m.Groups[2].Value}.");
        s = ResnetParts(s);
        s = TransformerParts(s);
        if (s == name || !IsUNetName(s))
        {
            throw new InvalidDataException($"Unexpected UNet tensor '{name}'.");
        }

        string canonical = "unet." + s;
        if (ProjectionPart().IsMatch(canonical))
        {
            weights.AddLinear(canonical, weight);
            return;
        }

        weights.Add(canonical, weight);
    }

    /// <summary>Adds one VAE tensor under its canonical <c>vae.*</c> name.</summary>
    /// <param name="name">Diffusers name (without a component prefix).</param>
    /// <param name="weight">The stored tensor.</param>
    /// <param name="weights">The canonical set.</param>
    /// <exception cref="InvalidDataException">The name is not part of a supported VAE.</exception>
    public static void AddVae(string name, WeightInfo weight, CanonicalWeights weights)
    {
        // FLUX.2's batch-norm statistics of the patchified latents (the step counter is not needed).
        if (name is "bn.running_mean" or "bn.running_var")
        {
            weights.AddF32("vae." + name, weight);
            return;
        }

        if (name == "bn.num_batches_tracked")
        {
            return;
        }

        string s = name.Replace("coder.conv_in.", "coder.input_conv.", StringComparison.Ordinal)
            .Replace("coder.conv_norm_out.", "coder.output_norm.", StringComparison.Ordinal)
            .Replace("coder.conv_out.", "coder.output_conv.", StringComparison.Ordinal);
        s = VaeLevelPart().Replace(s, "$1.$2.$3.res.$4.");
        s = VaeSampler().Replace(s, m => $"{m.Groups[1].Value}.{m.Groups[2].Value}.{m.Groups[3].Value}.{m.Groups[4].Value}sample.");
        s = VaeMidPart().Replace(s, m => $"{m.Groups[1].Value}.mid.{(m.Groups[2].Value == "resnets" ? "res" : "attn")}.{m.Groups[3].Value}.");
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
            string canonical = $"vae.{attention.Groups[1].Value}.{part}.{attention.Groups[3].Value}";
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

        if (!IsVaeName(s))
        {
            throw new InvalidDataException($"Unexpected VAE tensor '{name}'.");
        }

        weights.Add("vae." + s, weight);
    }

    /// <summary>
    /// Adds one tensor of a CLIP text encoder (Hugging Face <c>CLIPTextModel</c> names, with or without the <c>text_model.</c>
    /// prefix) under its canonical <c>{prefix}*</c> name; returns false for tensors that are not weights (position ids).
    /// </summary>
    /// <param name="name">Hugging Face name (without a component prefix).</param>
    /// <param name="weight">The stored tensor.</param>
    /// <param name="prefix">Canonical prefix of the encoder (<c>text_encoder.</c>, <c>text_encoder_2.</c>).</param>
    /// <param name="weights">The canonical set.</param>
    /// <exception cref="InvalidDataException">The name is not part of a CLIP text encoder.</exception>
    public static bool AddClipText(string name, WeightInfo weight, string prefix, CanonicalWeights weights)
    {
        string s = name.StartsWith("text_model.", StringComparison.Ordinal) ? name["text_model.".Length..] : name;
        if (s.EndsWith("position_ids", StringComparison.Ordinal))
        {
            return false;
        }

        switch (s)
        {
            case "embeddings.token_embedding.weight":
                weights.Add(prefix + "token_embedding.weight", weight);
                return true;
            case "embeddings.position_embedding.weight":
                weights.AddF32(prefix + "position_embedding.weight", weight);
                return true;
            case "final_layer_norm.weight" or "final_layer_norm.bias":
                weights.Add(prefix + "final_norm." + s[(s.LastIndexOf('.') + 1)..], weight);
                return true;
            case "text_projection.weight":
                weights.Add(prefix + "projection.weight", weight);
                return true;
        }

        var layer = ClipLayer().Match(s);
        if (!layer.Success)
        {
            throw new InvalidDataException($"Unexpected CLIP text encoder tensor '{name}'.");
        }

        string part = layer.Groups[2].Value switch
        {
            "layer_norm1" => "norm1",
            "layer_norm2" => "norm2",
            "self_attn.q_proj" => "attn.q",
            "self_attn.k_proj" => "attn.k",
            "self_attn.v_proj" => "attn.v",
            "self_attn.out_proj" => "attn.o",
            "mlp.fc1" => "mlp.up",
            _ => "mlp.down",
        };
        weights.Add($"{prefix}layers.{layer.Groups[1].Value}.{part}.{layer.Groups[3].Value}", weight);
        return true;
    }

    private static string Prefix(string s, string from, string to)
        => s.StartsWith(from, StringComparison.Ordinal) ? to + s[from.Length..] : s;

    private static string ResnetParts(string s)
        => s.Replace(".time_emb_proj.", ".time_proj.", StringComparison.Ordinal)
            .Replace(".conv_shortcut.", ".shortcut.", StringComparison.Ordinal);

    private static string TransformerParts(string s)
    {
        s = TransformerBlock().Replace(s, ".blocks.$1.");
        return s.Replace(".attn1.", ".self.", StringComparison.Ordinal)
            .Replace(".attn2.", ".cross.", StringComparison.Ordinal)
            .Replace(".to_q.", ".q.", StringComparison.Ordinal)
            .Replace(".to_k.", ".k.", StringComparison.Ordinal)
            .Replace(".to_v.", ".v.", StringComparison.Ordinal)
            .Replace(".to_out.0.", ".o.", StringComparison.Ordinal)
            .Replace(".ff.net.0.proj.", ".ff.proj.", StringComparison.Ordinal)
            .Replace(".ff.net.2.", ".ff.out.", StringComparison.Ordinal);
    }

    private static bool IsUNetName(string s) => s.StartsWith("time_embedding.", StringComparison.Ordinal)
        || s.StartsWith("class_embedding.", StringComparison.Ordinal) || s.StartsWith("input_conv.", StringComparison.Ordinal)
        || s.StartsWith("output_", StringComparison.Ordinal) || s.StartsWith("down.", StringComparison.Ordinal)
        || s.StartsWith("up.", StringComparison.Ordinal) || s.StartsWith("mid.", StringComparison.Ordinal);

    private static bool IsVaeName(string s) => s.StartsWith("encoder.", StringComparison.Ordinal)
        || s.StartsWith("decoder.", StringComparison.Ordinal) || s.StartsWith("quant_conv.", StringComparison.Ordinal)
        || s.StartsWith("post_quant_conv.", StringComparison.Ordinal);

    [GeneratedRegex(@"^(down|up)_blocks\.(\d+)\.(resnets|attentions)\.(\d+)\.")]
    private static partial Regex LevelPart();

    [GeneratedRegex(@"^down_blocks\.(\d+)\.downsamplers\.0\.conv\.")]
    private static partial Regex Downsampler();

    [GeneratedRegex(@"^up_blocks\.(\d+)\.upsamplers\.0\.conv\.")]
    private static partial Regex Upsampler();

    [GeneratedRegex(@"^mid_block\.(resnets|attentions)\.(\d+)\.")]
    private static partial Regex MidPart();

    [GeneratedRegex(@"\.transformer_blocks\.(\d+)\.")]
    private static partial Regex TransformerBlock();

    [GeneratedRegex(@"\.attn\.\d+\.proj_(in|out)\.weight$")]
    private static partial Regex ProjectionPart();

    [GeneratedRegex(@"^(encoder|decoder)\.(down|up)_blocks\.(\d+)\.resnets\.(\d+)\.")]
    private static partial Regex VaeLevelPart();

    [GeneratedRegex(@"^(encoder|decoder)\.(down|up)_blocks\.(\d+)\.(down|up)samplers\.0\.conv\.")]
    private static partial Regex VaeSampler();

    [GeneratedRegex(@"^(encoder|decoder)\.mid_block\.(resnets|attentions)\.(\d+)\.")]
    private static partial Regex VaeMidPart();

    [GeneratedRegex(@"^((?:encoder|decoder)\.mid\.attn\.0)\.(group_norm|to_q|to_k|to_v|to_out\.0|query|key|value|proj_attn)\.(weight|bias)$")]
    private static partial Regex VaeAttention();

    [GeneratedRegex(@"^encoder\.layers\.(\d+)\.(layer_norm1|layer_norm2|self_attn\.q_proj|self_attn\.k_proj|self_attn\.v_proj|self_attn\.out_proj|mlp\.fc1|mlp\.fc2)\.(weight|bias)$")]
    private static partial Regex ClipLayer();
}
