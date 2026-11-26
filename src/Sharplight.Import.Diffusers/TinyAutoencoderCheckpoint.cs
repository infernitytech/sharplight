using System.Globalization;
using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Tiny autoencoders (TAESD: madebyollin's taesd / taesdxl / taesd3 / taef1, diffusers' <c>AutoencoderTiny</c>): the diffusers file
/// (<c>encoder.layers.*</c>, <c>decoder.layers.*</c>, with its <c>config.json</c> beside it) or the original decoder / encoder files
/// (<c>taesd_decoder.safetensors</c>: the layers of an <c>nn.Sequential</c> numbered from 1 after its clamp; <c>taesd_encoder.safetensors</c>
/// from 0; a decoder file picks up the encoder file beside it). They open as sources of architecture <see cref="Architecture"/> with the
/// canonical weights <c>tiny_vae.{encoder|decoder}.layers.{i}</c> (convolutions; blocks <c>.conv1</c>, <c>.conv2</c>, <c>.conv3</c> and the
/// optional 1×1 <c>.skip</c>) and the metadata <c>tiny_autoencoder.*</c>, which image generators load as their <c>tiny-vae</c> auxiliary source.
/// </summary>
internal static partial class TinyAutoencoderCheckpoint
{
    /// <summary>The architecture id of tiny autoencoder sources.</summary>
    public const string Architecture = "tiny_autoencoder";

    private const string WeightPrefix = "tiny_vae.";
    private const string MetadataPrefix = Architecture + ".";

    /// <summary>Whether the tensor names are a tiny autoencoder (diffusers names, or an original decoder or encoder file).</summary>
    /// <param name="names">The tensor names.</param>
    public static bool IsTinyAutoencoder(IReadOnlyCollection<string> names)
    {
        if (names.Count == 0)
        {
            return false;
        }

        bool diffusers = names.Contains("decoder.layers.0.weight") || names.Contains("encoder.layers.0.weight");
        bool original = (names.Contains("1.weight") && names.Contains("3.conv.0.weight")) || (names.Contains("0.weight") && names.Contains("1.conv.0.weight"));
        return (diffusers || original) && names.All(n => OriginalName().IsMatch(n) || DiffusersName().IsMatch(n));
    }

    /// <summary>Opens the file as a tiny autoencoder source.</summary>
    /// <param name="path">The safetensors file.</param>
    /// <exception cref="NotSupportedException">The file is not a tiny autoencoder.</exception>
    public static ModelSource Open(string path)
    {
        var tensors = new CheckpointTensors();
        var sibling = new CheckpointTensors();
        try
        {
            tensors.Add(path);
            var names = tensors.Tensors.Keys.ToList();
            if (!IsTinyAutoencoder(names))
            {
                throw new NotSupportedException($"'{path}' is not a tiny autoencoder (TAESD).");
            }

            var weights = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
            if (names.Any(n => n.StartsWith("decoder.", StringComparison.Ordinal) || n.StartsWith("encoder.", StringComparison.Ordinal)))
            {
                foreach (var (name, weight) in tensors.Tensors)
                {
                    Add(weights, name, weight, 0);
                }
            }
            else
            {
                bool decoder = names.Contains("1.weight");
                AddOriginal(weights, tensors.Tensors, decoder);
                string siblingPath = SiblingFile(path, decoder);
                if (File.Exists(siblingPath))
                {
                    sibling.Add(siblingPath);
                    AddOriginal(weights, sibling.Tensors, !decoder);
                }
            }

            string configPath = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(path))!, "config.json");
            JsonElement? config = File.Exists(configPath) ? ReadJson(configPath) : null;
            if (config is { } c && c.String("_class_name", "") != "AutoencoderTiny")
            {
                config = null;
            }

            var metadata = new Dictionary<string, object>(StringComparer.Ordinal)
            {
                [CanonicalKeys.Name] = Path.GetFileNameWithoutExtension(path),
                [CanonicalKeys.Architecture] = Architecture,
            };
            WriteConfig(weights, config, metadata);
            return new ModelSource
            {
                Format = "safetensors",
                Architecture = Architecture,
                Metadata = metadata,
                Weights = weights,
                OwnedResources = [.. tensors.Resources, .. sibling.Resources],
                Path = path,
            };
        }
        catch
        {
            tensors.DisposeAll();
            sibling.DisposeAll();
            throw;
        }
    }

    /// <summary>The encoder file beside a decoder file, or the decoder beside an encoder (<c>taesd_decoder</c> / <c>taesd_encoder</c>).</summary>
    private static string SiblingFile(string path, bool decoder)
    {
        string name = Path.GetFileName(path);
        string replaced = decoder ? name.Replace("_decoder", "_encoder", StringComparison.Ordinal) : name.Replace("_encoder", "_decoder", StringComparison.Ordinal);
        return replaced == name ? "" : Path.Combine(Path.GetDirectoryName(Path.GetFullPath(path))!, replaced);
    }

    private static void AddOriginal(Dictionary<string, WeightInfo> weights, IReadOnlyDictionary<string, WeightInfo> tensors, bool decoder)
    {
        // The original decoder starts with a parameter-free clamp at index 0; diffusers' layer list (the canonical numbering) does not.
        foreach (var (name, weight) in tensors)
        {
            Add(weights, (decoder ? "decoder.layers." : "encoder.layers.") + name, weight, decoder ? 1 : 0);
        }
    }

    private static void Add(Dictionary<string, WeightInfo> weights, string name, WeightInfo weight, int offset)
    {
        var match = DiffusersName().Match(name);
        if (!match.Success)
        {
            throw new InvalidDataException($"Unexpected tiny autoencoder tensor '{name}'.");
        }

        int layer = int.Parse(match.Groups[2].Value, CultureInfo.InvariantCulture) - offset;
        string part = match.Groups[3].Value switch
        {
            "" => "",
            ".conv.0" => ".conv1",
            ".conv.2" => ".conv2",
            ".conv.4" => ".conv3",
            ".skip" => ".skip",
            var other => throw new InvalidDataException($"Unexpected tiny autoencoder tensor '{name}' ({other})."),
        };
        string canonical = string.Create(CultureInfo.InvariantCulture, $"{WeightPrefix}{match.Groups[1].Value}.layers.{layer}{part}.{match.Groups[4].Value}");
        weights[canonical] = weight.As(canonical);
    }

    /// <summary>
    /// The configuration: from <c>config.json</c> when there is one, otherwise inferred from the layers (the published models: 64 channels
    /// per level, encoder blocks 1 / 3 / 3 / 3, decoder blocks 3 / 3 / 3 / 1, no latent scaling).
    /// </summary>
    private static void WriteConfig(Dictionary<string, WeightInfo> weights, JsonElement? config, Dictionary<string, object> metadata)
    {
        int latentChannels = weights.TryGetValue(WeightPrefix + "decoder.layers.0.weight", out var first)
            ? (int)first.Shape[1].Fixed
            : (int)weights[EncoderOutput(weights)].Shape[0].Fixed;
        int[] decoderBlocks = config?.Ints("num_decoder_blocks") ?? [3, 3, 3, 1];
        int[] encoderBlocks = config?.Ints("num_encoder_blocks") ?? [1, 3, 3, 3];
        int[] decoderChannels = config is { } c && c.TryGetProperty("decoder_block_out_channels", out _) ? c.Ints("decoder_block_out_channels") : [64, 64, 64, 64];
        int[] encoderChannels = config is { } e && e.TryGetProperty("encoder_block_out_channels", out _) ? e.Ints("encoder_block_out_channels") : [64, 64, 64, 64];
        if (config is { } check && (check.String("act_fn", "relu") != "relu" || check.String("upsample_fn", "nearest") != "nearest"))
        {
            throw new NotSupportedException($"Not supported yet: tiny autoencoders with act_fn '{check.String("act_fn", "")}' / upsample_fn '{check.String("upsample_fn", "")}'.");
        }

        metadata[MetadataPrefix + "latent_channels"] = latentChannels;
        metadata[MetadataPrefix + "decoder_blocks"] = decoderBlocks;
        metadata[MetadataPrefix + "encoder_blocks"] = encoderBlocks;
        metadata[MetadataPrefix + "decoder_channels"] = decoderChannels;
        metadata[MetadataPrefix + "encoder_channels"] = encoderChannels;
        metadata[MetadataPrefix + "upsample_factor"] = config?.Int("upsampling_scaling_factor", 2) ?? 2;
        metadata[MetadataPrefix + "scaling_factor"] = config?.Float("scaling_factor", 1f) ?? 1f;
        metadata[MetadataPrefix + "shift_factor"] = config?.Float("shift_factor", 0f) ?? 0f;
        metadata[MetadataPrefix + "has_decoder"] = weights.ContainsKey(WeightPrefix + "decoder.layers.0.weight");
        metadata[MetadataPrefix + "has_encoder"] = weights.ContainsKey(WeightPrefix + "encoder.layers.0.weight");
    }

    /// <summary>The encoder's last convolution (the highest-numbered plain layer).</summary>
    private static string EncoderOutput(Dictionary<string, WeightInfo> weights)
        => weights.Keys.Where(k => k.StartsWith(WeightPrefix + "encoder.layers.", StringComparison.Ordinal) && k.EndsWith(".weight", StringComparison.Ordinal)
                && k.Count(ch => ch == '.') == 4)
            .OrderBy(k => int.Parse(k.Split('.')[3], CultureInfo.InvariantCulture)).Last();

    private static JsonElement ReadJson(string path)
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(path));
        return document.RootElement.Clone();
    }

    [GeneratedRegex(@"^(encoder|decoder)\.layers\.(\d+)((?:\.conv\.\d+|\.skip)?)\.(weight|bias)$")]
    private static partial Regex DiffusersName();

    [GeneratedRegex(@"^\d+(\.conv\.\d+|\.skip)?\.(weight|bias)$")]
    private static partial Regex OriginalName();
}
