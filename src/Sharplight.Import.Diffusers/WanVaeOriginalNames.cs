using System.Globalization;
using System.Text.RegularExpressions;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// The original names of a Wan 2.1-layout causal 3-D VAE (the Wan repository's and ComfyUI's files, e.g. <c>qwen_image_vae.safetensors</c>:
/// <c>encoder.conv1</c>, <c>encoder.downsamples.{i}.residual.{0,2,3,6}</c>, <c>decoder.upsamples.{i}</c>, <c>*.middle.{0,1,2}</c>,
/// <c>*.head.{0,2}</c>, <c>conv1</c> / <c>conv2</c> for the quant convolutions) mapped to diffusers' <c>AutoencoderKLWan</c> names
/// (as diffusers' <c>convert_wan_vae_to_diffusers</c>, with the decoder's per-level block counts taken from the file).
/// </summary>
internal static partial class WanVaeOriginalNames
{
    private static readonly (string Original, string Diffusers)[] s_residualParts =
    [
        (".residual.0.gamma", ".norm1.gamma"), (".residual.2.", ".conv1."), (".residual.3.gamma", ".norm2.gamma"), (".residual.6.", ".conv2."),
        (".shortcut.", ".conv_shortcut."),
    ];

    /// <summary>Whether the tensor names are an original-layout Wan VAE.</summary>
    /// <param name="names">The tensor names.</param>
    public static bool IsOriginal(IReadOnlyCollection<string> names)
        => names.Contains("encoder.conv1.weight") && names.Contains("decoder.head.0.gamma") && names.Contains("conv2.weight");

    /// <summary>Whether the tensor names are a diffusers <c>AutoencoderKLWan</c> / <c>AutoencoderKLQwenImage</c>.</summary>
    /// <param name="names">The tensor names.</param>
    public static bool IsDiffusers(IReadOnlyCollection<string> names)
        => names.Contains("encoder.conv_in.weight") && names.Contains("decoder.norm_out.gamma") && names.Contains("post_quant_conv.weight");

    /// <summary>The diffusers name of every original tensor name.</summary>
    /// <param name="names">The original tensor names.</param>
    /// <exception cref="InvalidDataException">A name has no diffusers counterpart.</exception>
    public static Dictionary<string, string> Map(IReadOnlyCollection<string> names)
    {
        // Decoder levels: residual blocks, then (except the last level) one resampler; the first level's count gives the block count.
        int blocks = 0;
        while (names.Contains($"decoder.upsamples.{blocks}.residual.0.gamma"))
        {
            blocks++;
        }

        var result = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (string name in names)
        {
            result[name] = DiffusersName(name, blocks) ?? throw new InvalidDataException($"Unexpected Wan VAE tensor '{name}'.");
        }

        return result;
    }

    private static string? DiffusersName(string name, int decoderBlocks)
    {
        switch (name)
        {
            case "conv1.weight" or "conv1.bias":
                return "quant_conv" + name[5..];
            case "conv2.weight" or "conv2.bias":
                return "post_quant_conv" + name[5..];
        }

        var m = Part().Match(name);
        if (!m.Success)
        {
            return null;
        }

        string side = m.Groups[1].Value;
        string module = m.Groups[2].Value;
        string rest = m.Groups[3].Value;
        switch (module)
        {
            case "conv1":
                return $"{side}.conv_in{rest}";
            case "head":
                return rest.StartsWith(".0.", StringComparison.Ordinal) ? $"{side}.norm_out{rest[2..]}"
                    : rest.StartsWith(".2.", StringComparison.Ordinal) ? $"{side}.conv_out{rest[2..]}"
                    : null;
            case "middle":
            {
                int index = rest[1] - '0';
                string tail = rest[2..];
                return index == 1 ? $"{side}.mid_block.attentions.0{tail}" : Residual($"{side}.mid_block.resnets.{index / 2}", tail);
            }

            case "downsamples":
            {
                int dot = rest.IndexOf('.', 1);
                string block = rest[1..dot];
                string tail = rest[dot..];
                return tail.StartsWith(".resample.", StringComparison.Ordinal) || tail.StartsWith(".time_conv.", StringComparison.Ordinal)
                    ? $"{side}.down_blocks.{block}{tail}"
                    : Residual($"{side}.down_blocks.{block}", tail);
            }

            case "upsamples":
            {
                int dot = rest.IndexOf('.', 1);
                int block = int.Parse(rest[1..dot], CultureInfo.InvariantCulture);
                string tail = rest[dot..];
                int stride = decoderBlocks + 1;
                int level = block / stride;
                int within = block % stride;
                if (within == decoderBlocks)
                {
                    return tail.StartsWith(".resample.", StringComparison.Ordinal) || tail.StartsWith(".time_conv.", StringComparison.Ordinal)
                        ? $"{side}.up_blocks.{level}.upsamplers.0{tail}"
                        : null;
                }

                return Residual($"{side}.up_blocks.{level}.resnets.{within}", tail);
            }

            default:
                return null;
        }
    }

    private static string? Residual(string block, string tail)
    {
        foreach (var (original, diffusers) in s_residualParts)
        {
            if (tail.StartsWith(original, StringComparison.Ordinal))
            {
                return block + diffusers + tail[original.Length..];
            }
        }

        return null;
    }

    [GeneratedRegex(@"^(encoder|decoder)\.(conv1|head|middle|downsamples|upsamples)(\..*)$")]
    private static partial Regex Part();
}
