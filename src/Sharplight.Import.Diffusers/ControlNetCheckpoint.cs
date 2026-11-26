using System.Globalization;
using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// ControlNets for the SD 1.x / 2.x / XL UNets in diffusers' <c>ControlNetModel</c> layout (lllyasviel/control_v11p_sd15_canny,
/// diffusers/controlnet-canny-sdxl-1.0, ...: <c>diffusion_pytorch_model.safetensors</c> with its <c>config.json</c> beside it). They open as
/// sources of architecture <see cref="Architecture"/>: the UNet half under <c>controlnet.</c> with the UNet's canonical names
/// (<c>controlnet.input_conv</c>, <c>controlnet.time_embedding.*</c>, <c>controlnet.class_embedding.*</c>, <c>controlnet.down.*</c>,
/// <c>controlnet.mid.*</c>), the control image's embedding (<c>controlnet.condition_embedding.{input_conv, blocks.{i}, output_conv}</c>) and the
/// zero convolutions (<c>controlnet.zero_convs.{i}</c>, <c>controlnet.mid_zero_conv</c>); metadata <c>controlnet.unet.*</c> (the UNet keys),
/// <c>controlnet.condition_channels</c> and <c>controlnet.condition_embedding_channels</c>. Image generators load them as their
/// <c>controlnet</c> auxiliary source. The original release files (lllyasviel's <c>control_v11p_sd15_canny.pth</c>, their <c>.safetensors</c>
/// conversions: LDM names under <c>control_model.</c>) map to the same names, their configuration inferred from the shapes.
/// </summary>
internal static partial class ControlNetCheckpoint
{
    /// <summary>The architecture id of ControlNet sources.</summary>
    public const string Architecture = "controlnet";

    private const string Prefix = "controlnet.";

    // The original release files (lllyasviel/ControlNet-v1-1 .pth, ComfyUI's .safetensors conversions): LDM names under control_model.
    private const string OriginalPrefix = "control_model.";

    /// <summary>Whether the tensor names are a diffusers ControlNet.</summary>
    /// <param name="names">The tensor names.</param>
    public static bool IsControlNet(IReadOnlyCollection<string> names)
        => (names.Contains("controlnet_cond_embedding.conv_in.weight") && names.Contains("controlnet_mid_block.weight"))
           || (names.Contains(OriginalPrefix + "input_hint_block.0.weight") && names.Contains(OriginalPrefix + "middle_block_out.0.weight"));

    /// <summary>Opens the file as a ControlNet source.</summary>
    /// <param name="path">The safetensors file (its <c>config.json</c> beside it).</param>
    /// <exception cref="NotSupportedException">The file is not a supported ControlNet.</exception>
    public static ModelSource Open(string path)
    {
        var tensors = new CheckpointTensors();
        try
        {
            tensors.Add(path);
            if (!IsControlNet([.. tensors.Tensors.Keys]))
            {
                throw new NotSupportedException($"'{path}' is not a ControlNet.");
            }

            if (tensors.Tensors.ContainsKey(OriginalPrefix + "input_hint_block.0.weight"))
            {
                return OpenOriginal(path, tensors);
            }

            string configPath = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(path))!, "config.json");
            if (!File.Exists(configPath))
            {
                throw new NotSupportedException($"'{path}': a ControlNet in diffusers' layout needs the config.json of its folder beside it.");
            }

            JsonElement config;
            using (var document = JsonDocument.Parse(File.ReadAllBytes(configPath)))
            {
                config = document.RootElement.Clone();
            }

            Validate(config, path);
            var unet = new CanonicalWeights();
            var weights = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
            foreach (var (name, weight) in tensors.Tensors)
            {
                if (ExtraName(name) is { } extra)
                {
                    var single = new CanonicalWeights();
                    single.Add(Prefix + extra, weight);
                    foreach (var (canonical, added) in single.Weights)
                    {
                        weights[canonical] = added;
                    }

                    continue;
                }

                DiffusersNames.AddUNet(name, weight, unet);
            }

            foreach (var (name, weight) in unet.Weights)
            {
                string canonical = Prefix + name["unet.".Length..];
                weights[canonical] = weight.As(canonical);
            }

            var metadata = new Dictionary<string, object>(StringComparer.Ordinal)
            {
                [CanonicalKeys.Name] = Path.GetFileName(Path.GetDirectoryName(Path.GetFullPath(path))!),
                [CanonicalKeys.Architecture] = Architecture,
                [Prefix + "condition_channels"] = config.Int("conditioning_channels", 3),
                [Prefix + "condition_embedding_channels"] = config.TryGetProperty("conditioning_embedding_out_channels", out var embedding)
                    && embedding.ValueKind == JsonValueKind.Array ? config.Ints("conditioning_embedding_out_channels") : new[] { 16, 32, 96, 256 },
            };
            UNetConfig.FromDiffusers(config).WriteTo(metadata, Prefix);
            return Source(path, metadata, weights, tensors);
        }
        catch
        {
            tensors.DisposeAll();
            throw;
        }
    }

    /// <summary>
    /// An original-layout ControlNet: the UNet half in LDM names (<see cref="LdmNames"/>), the hint block's convolutions
    /// (<c>input_hint_block.{0, 2, ..., 14}</c>: input, the six blocks, output), <c>zero_convs.{i}.0</c> and <c>middle_block_out.0</c>; the
    /// configuration inferred from the shapes (as for single-file checkpoints).
    /// </summary>
    private static ModelSource OpenOriginal(string path, CheckpointTensors tensors)
    {
        var unetNames = tensors.Tensors.Keys.Where(n => n.StartsWith(OriginalPrefix, StringComparison.Ordinal)).Select(n => n[OriginalPrefix.Length..]).ToList();
        var unet = new CanonicalWeights();
        foreach (var (ldm, diffusers) in LdmNames.UNet(unetNames))
        {
            DiffusersNames.AddUNet(diffusers, tensors.Tensors[OriginalPrefix + ldm], unet);
        }

        var weights = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
        var extras = new CanonicalWeights();
        int hintConvs = 0;
        foreach (string name in unetNames)
        {
            var hint = HintConv().Match(name);
            var zero = OriginalZeroConv().Match(name);
            string? canonical = hint.Success ? HintName(int.Parse(hint.Groups[1].Value, CultureInfo.InvariantCulture), hint.Groups[2].Value)
                : zero.Success ? $"zero_convs.{zero.Groups[1].Value}.{zero.Groups[2].Value}"
                : name.StartsWith("middle_block_out.0.", StringComparison.Ordinal) ? "mid_zero_conv." + name["middle_block_out.0.".Length..]
                : null;
            if (canonical is not null)
            {
                extras.Add(Prefix + canonical, tensors.Tensors[OriginalPrefix + name]);
                hintConvs += hint.Success && hint.Groups[2].Value == "weight" ? 1 : 0;
            }
        }

        foreach (var (name, weight) in extras.Weights)
        {
            weights[name] = weight;
        }

        foreach (var (name, weight) in unet.Weights)
        {
            string canonical = Prefix + name["unet.".Length..];
            weights[canonical] = weight.As(canonical);
        }

        // Embedding levels: the input convolution's width, then every second block's (the strided ones).
        int levels = (hintConvs - 2) / 2 + 1;
        int[] embedding = [(int)weights[Prefix + "condition_embedding.input_conv.weight"].Shape[0].Fixed,
            .. Enumerable.Range(0, levels - 1).Select(i => (int)weights[$"{Prefix}condition_embedding.blocks.{2 * i + 1}.weight"].Shape[0].Fixed)];
        bool xl = unet.Contains("unet.class_embedding.linear1.weight");
        var metadata = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [CanonicalKeys.Name] = Path.GetFileNameWithoutExtension(path),
            [CanonicalKeys.Architecture] = Architecture,
            [Prefix + "condition_channels"] = (int)weights[Prefix + "condition_embedding.input_conv.weight"].Shape[1].Fixed,
            [Prefix + "condition_embedding_channels"] = embedding,
        };
        UNetConfig.FromWeights(unet.Weights, xl ? 128 : 64).WriteTo(metadata, Prefix);
        return Source(path, metadata, weights, tensors);
    }

    /// <summary>The canonical name (after <c>controlnet.condition_embedding.</c>) of hint block convolution <paramref name="index"/> (0, 2, ..., 14).</summary>
    private static string HintName(int index, string part)
    {
        int last = 14;
        string conv = index == 0 ? "input_conv" : index == last ? "output_conv" : string.Create(CultureInfo.InvariantCulture, $"blocks.{index / 2 - 1}");
        return $"condition_embedding.{conv}.{part}";
    }

    private static ModelSource Source(string path, Dictionary<string, object> metadata, Dictionary<string, WeightInfo> weights, CheckpointTensors tensors)
        => new()
        {
            Format = Path.GetExtension(path).Equals(".safetensors", StringComparison.OrdinalIgnoreCase) ? "safetensors" : "pytorch",
            Architecture = Architecture,
            Metadata = metadata,
            Weights = weights,
            OwnedResources = tensors.Resources,
            Path = path,
        };

    /// <summary>The canonical name (after <c>controlnet.</c>) of a ControlNet-only tensor, or null for a tensor of the UNet half.</summary>
    private static string? ExtraName(string name)
    {
        if (name.StartsWith("controlnet_cond_embedding.", StringComparison.Ordinal))
        {
            return "condition_embedding." + name["controlnet_cond_embedding.".Length..]
                .Replace("conv_in.", "input_conv.", StringComparison.Ordinal)
                .Replace("conv_out.", "output_conv.", StringComparison.Ordinal);
        }

        var down = ZeroConv().Match(name);
        if (down.Success)
        {
            return string.Create(CultureInfo.InvariantCulture, $"zero_convs.{down.Groups[1].Value}.{down.Groups[2].Value}");
        }

        return name.StartsWith("controlnet_mid_block.", StringComparison.Ordinal) ? "mid_zero_conv." + name["controlnet_mid_block.".Length..] : null;
    }

    private static void Validate(JsonElement config, string path)
    {
        string? unsupported = config.String("controlnet_conditioning_channel_order", "rgb") != "rgb" ? "BGR control images"
            : config.TryGetProperty("encoder_hid_dim", out var hid) && hid.ValueKind != JsonValueKind.Null ? "a projected text conditioning (encoder_hid_dim)"
            : config.TryGetProperty("class_embed_type", out var classEmbed) && classEmbed.ValueKind != JsonValueKind.Null ? $"class embeddings ({classEmbed})"
            : config.String("addition_embed_type", "text_time") is not ("text_time" or "") ? $"the addition embedding '{config.String("addition_embed_type", "")}'"
            : null;
        if (unsupported is not null)
        {
            throw new NotSupportedException($"Not supported yet: ControlNets with {unsupported} ('{path}').");
        }
    }

    [GeneratedRegex(@"^controlnet_down_blocks\.(\d+)\.(weight|bias)$")]
    private static partial Regex ZeroConv();

    [GeneratedRegex(@"^input_hint_block\.(\d+)\.(weight|bias)$")]
    private static partial Regex HintConv();

    [GeneratedRegex(@"^zero_convs\.(\d+)\.0\.(weight|bias)$")]
    private static partial Regex OriginalZeroConv();
}
