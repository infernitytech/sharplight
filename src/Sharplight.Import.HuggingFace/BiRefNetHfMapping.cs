using System.Globalization;
using System.Text.RegularExpressions;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Vision;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// BiRefNet and its derivatives (<c>ZhengPeng7/BiRefNet</c>, <c>briaai/RMBG-2.0</c>: remote-code checkpoints whose <c>config.json</c> only names the class <c>BiRefNet</c>) → canonical
/// <c>birefnet</c> (documented on <c>BiRefNetAdapter</c>). The network's hyper-parameters are not in the configuration: they are the constants of the checkpoint's own
/// <c>birefnet.py</c>, which is read as text (never run) for the backbone (Swin tiny, small, base or large) and checked for the supported decoder setup. Batch norms are folded into their
/// convolutions. The processor is the checkpoint's <c>preprocessor_config.json</c>, else a 1024 px square resize with ImageNet statistics.
/// </summary>
internal sealed class BiRefNetHfMapping : IHfArchitectureMapping
{
    private const string ModelTypeName = "birefnet";

    // Backbones: embed dimension, depths, heads, window.
    private static readonly Dictionary<string, (int Embed, int[] Depths, int[] Heads, int Window)> s_backbones = new(StringComparer.Ordinal)
    {
        ["swin_v1_t"] = (96, [2, 2, 6, 2], [3, 6, 12, 24], 7),
        ["swin_v1_s"] = (96, [2, 2, 18, 2], [3, 6, 12, 24], 7),
        ["swin_v1_b"] = (128, [2, 2, 18, 2], [4, 8, 16, 32], 12),
        ["swin_v1_l"] = (192, [2, 2, 18, 2], [6, 12, 24, 48], 12),
    };

    // The settings of birefnet.py the adapter implements.
    private static readonly (string Name, string Expected)[] s_required =
    [
        ("mul_scl_ipt", "cat"),
        ("dec_att", "ASPPDeformable"),
        ("squeeze_block", "BasicDecBlk_x1"),
        ("dec_blk", "BasicDecBlk"),
        ("lat_blk", "BasicLatBlk"),
        ("refine", ""),
    ];

    /// <inheritdoc/>
    public string Architecture => ModelTypeName;

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = [ModelTypeName];

    /// <inheritdoc/>
    public bool NeedsTokenizer => false;

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig c)
    {
        string? source = Source(c);
        if (source is null)
        {
            yield return "birefnet: the checkpoint has no birefnet.py to read the network's settings from.";
            yield break;
        }

        string? backbone = Choice(source, "bb");
        if (backbone is null || !s_backbones.ContainsKey(backbone))
        {
            yield return $"birefnet: the backbone '{backbone}' is not supported (swin_v1_t, swin_v1_s, swin_v1_b, swin_v1_l).";
        }

        foreach (var (name, expected) in s_required)
        {
            string? value = Choice(source, name);
            if (value != expected)
            {
                yield return $"birefnet: {name} '{value}' is not supported (only '{expected}').";
            }
        }

        if (Choice(source, "cxt_num") != "3")
        {
            yield return "birefnet: only the three multi-scale skip connections (cxt_num 3) are supported.";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        var backbone = s_backbones[Choice(Source(c)!, "bb")!];
        string path = Path.Combine(c.Directory, "preprocessor_config.json");
        var processor = File.Exists(path) ? HfImageProcessorConfig.Read(c.Directory, 1024) : new ImageProcessorConfig
        {
            ResizeMode = ImageResizeMode.Exact,
            Width = 1024,
            Height = 1024,
            Filter = ImageResampleFilter.Bilinear,
            Mean = [0.485f, 0.456f, 0.406f],
            Std = [0.229f, 0.224f, 0.225f],
        };
        string a = Architecture + ".";
        return new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "image_size"] = HfImageProcessorConfig.InputSize(processor, 1024),
            [a + "embed_dim"] = backbone.Embed,
            [a + "depths"] = string.Join(',', backbone.Depths),
            [a + "head_counts"] = string.Join(',', backbone.Heads),
            [a + "window"] = backbone.Window,
            [a + "aspp_kernels"] = "1,3,7",
            [a + "patch_order"] = Source(c)!.Contains("get_patches_batch", StringComparison.Ordinal) ? "tile" : "channel",
            [CanonicalKeys.ImageProcessor] = processor.ToJson().ToJsonString(),
        };
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        var collector = new HfWeightCollector();
        var normPrefixes = tensors.Keys.Where(k => k.EndsWith(".running_mean", StringComparison.Ordinal)).Select(k => k[..^".running_mean".Length]).ToHashSet(StringComparer.Ordinal);
        var consumedBiases = new HashSet<string>(StringComparer.Ordinal);
        foreach (var (rawName, tensor) in tensors)
        {
            if (rawName.EndsWith("relative_position_index", StringComparison.Ordinal) || rawName.EndsWith("num_batches_tracked", StringComparison.Ordinal)
                || normPrefixes.Contains(rawName[..Math.Max(0, rawName.LastIndexOf('.'))]))
            {
                continue;
            }

            string? name = Rename(rawName);
            if (name is null)
            {
                collector.AddUnmapped(rawName);
                continue;
            }

            if (rawName.EndsWith(".weight", StringComparison.Ordinal) && NormOf(rawName[..^".weight".Length], normPrefixes) is { } norm)
            {
                string convBias = rawName[..^".weight".Length] + ".bias";
                SafetensorsTensor T(string suffix) => tensors[norm + suffix];
                var (w, b) = HfBatchNormFolding.Fold(tensor.ToWeightInfo(), T(".weight").ToWeightInfo(), T(".bias").ToWeightInfo(), T(".running_mean").ToWeightInfo(),
                    T(".running_var").ToWeightInfo(), 1e-5f, name[..^".weight".Length], tensors.TryGetValue(convBias, out var own) ? own.ToWeightInfo() : null);
                collector.Add(w);
                collector.Add(b);
                consumedBiases.Add(convBias);
                continue;
            }

            if (consumedBiases.Contains(rawName) || (rawName.EndsWith(".bias", StringComparison.Ordinal) && NormOf(rawName[..^".bias".Length], normPrefixes) is not null))
            {
                continue;
            }

            collector.Add(HfTransforms.ToF32(tensor.ToWeightInfo(), name));
        }

        collector.ThrowIfUnmapped(c.ModelType);
        return collector.Weights;
    }

    private static string? Rename(string name)
    {
        if (name.StartsWith("bb.", StringComparison.Ordinal))
        {
            return "enc." + name[3..];
        }

        if (name.StartsWith("decoder.", StringComparison.Ordinal))
        {
            return "dec." + name[8..];
        }

        return name.StartsWith("squeeze_module.0.", StringComparison.Ordinal) ? "squeeze." + name["squeeze_module.0.".Length..] : null;
    }

    // The batch norm that follows the convolution with this weight prefix, if the checkpoint has one.
    private static string? NormOf(string convPrefix, HashSet<string> norms)
    {
        string? candidate = null;
        if (convPrefix.EndsWith(".conv_in", StringComparison.Ordinal))
        {
            candidate = convPrefix[..^".conv_in".Length] + ".bn_in";
        }
        else if (convPrefix.EndsWith(".conv_out", StringComparison.Ordinal))
        {
            candidate = convPrefix[..^".conv_out".Length] + ".bn_out";
        }
        else if (convPrefix.EndsWith(".atrous_conv.regular_conv", StringComparison.Ordinal))
        {
            candidate = convPrefix[..^".atrous_conv.regular_conv".Length] + ".bn";
        }
        else if (convPrefix.EndsWith(".dec_att.conv1", StringComparison.Ordinal))
        {
            candidate = convPrefix[..^".conv1".Length] + ".bn1";
        }
        else if (convPrefix.EndsWith(".global_avg_pool.1", StringComparison.Ordinal))
        {
            candidate = convPrefix[..^".1".Length] + ".2";
        }
        else if (convPrefix.Contains("gdt_convs_", StringComparison.Ordinal) && convPrefix.EndsWith(".0", StringComparison.Ordinal))
        {
            candidate = convPrefix[..^".0".Length] + ".1";
        }

        return candidate is not null && norms.Contains(candidate) ? candidate : null;
    }

    private static string? Source(HfConfig c)
    {
        string path = Path.Combine(c.Directory, "birefnet.py");
        return File.Exists(path) ? File.ReadAllText(path) : null;
    }

    // The selected entry of the source's "self.name = [a, b, c][index]" its value, unquoted; a plain assignment's text otherwise.
    private static string? Choice(string source, string name)
    {
        var indexed = Regex.Match(source, @"self\." + name + @"\s*=\s*\[(?<items>[^\]]*)\]\[(?<index>\d+)\]");
        if (indexed.Success)
        {
            var items = Regex.Replace(indexed.Groups["items"].Value, @"#[^\r\n]*", "").Split(',').Select(s => s.Trim().Trim('\'', '"')).ToArray();
            int index = int.Parse(indexed.Groups["index"].Value, CultureInfo.InvariantCulture);
            return index < items.Length ? items[index] : null;
        }

        var plain = Regex.Match(source, @"self\." + name + @"\s*=\s*(?<value>[^\r\n#]*)");
        return plain.Success ? plain.Groups["value"].Value.Trim().Trim('\'', '"') : null;
    }
}
