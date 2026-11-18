using System.Globalization;
using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Tensors;
using Sharplight.Vision;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// timm MobileNetV3 and EfficientNet / EfficientNetV2 checkpoints (<c>timm/mobilenetv3_small_100.lamb_in1k</c>, <c>timm/efficientnet_b0.ra_in1k</c>, <c>timm/tf_efficientnetv2_s.in21k_ft_in1k</c>: reported by
/// <see cref="HfConfig.ModelType"/> as <c>timm_mbconv</c>) → canonical <c>mbconvnet</c> (documented on <c>MbConvClassifierAdapter</c>). The block kinds follow from the weights of each block; the strides,
/// activations and padding style are per architecture family (the stride sits on the first block of a stage). Batch norms are folded into the convolutions. The processor follows
/// <c>pretrained_cfg</c> with timm's rounded crop offsets. Only the architectures listed in <c>s_families</c> are mapped.
/// </summary>
internal sealed partial class MbConvHfMapping : IHfArchitectureMapping
{
    private sealed record Family(int[] Strides, string[] Activations, string StemActivation, string SeActivation, string SeGate, string HeadActivation, bool PoolFirst, bool Same, float Eps, bool ConvSkip);

    private static readonly Family s_mobileNetSmall = new([2, 2, 2, 1, 2, 1], ["relu", "relu", "hswish", "hswish", "hswish", "hswish"], "hswish", "relu", "hsigmoid", "hswish", true, false, 1e-5f, false);

    private static readonly Family s_mobileNetLarge = new([1, 2, 2, 2, 1, 2, 1], ["relu", "relu", "relu", "hswish", "hswish", "hswish", "hswish"], "hswish", "relu", "hsigmoid", "hswish", true, false, 1e-5f, false);

    private static readonly Family s_efficientNet = new([1, 2, 2, 2, 1, 2, 1], ["silu", "silu", "silu", "silu", "silu", "silu", "silu"], "silu", "silu", "sigmoid", "silu", false, false, 1e-5f, false);

    private static readonly Family s_efficientNetV2S = new([1, 2, 2, 2, 1, 2], ["silu", "silu", "silu", "silu", "silu", "silu"], "silu", "silu", "sigmoid", "silu", false, true, 1e-3f, true);

    /// <inheritdoc/>
    public string Architecture => "mbconvnet";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = [HfConfig.TimmMbConvModelType];

    /// <inheritdoc/>
    public bool NeedsTokenizer => false;

    /// <summary>Whether timm's architecture name is one of the mapped families.</summary>
    public static bool Supports(string name) => FamilyOf(name) is not null;

    private static Family? FamilyOf(string name) => name switch
    {
        "mobilenetv3_small_050" or "mobilenetv3_small_075" or "mobilenetv3_small_100" => s_mobileNetSmall,
        "mobilenetv3_large_075" or "mobilenetv3_large_100" => s_mobileNetLarge,
        "efficientnet_b0" or "efficientnet_b1" or "efficientnet_b2" or "efficientnet_b3" or "efficientnet_b4" => s_efficientNet,
        "tf_efficientnetv2_s" => s_efficientNetV2S,
        _ => null,
    };

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig c)
    {
        string name = c.GetString("architecture") ?? "";
        if (FamilyOf(name) is null)
        {
            yield return $"timm: the architecture '{name}' is not supported (mobilenetv3_small_050 / 075 / 100, mobilenetv3_large_075 / 100, efficientnet_b0 to b4, tf_efficientnetv2_s).";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        var family = FamilyOf(c.GetString("architecture")!)!;
        var processor = TimmProcessor(c);
        var kinds = BlockKinds(c.Directory);
        var blocks = new List<string>();
        foreach (var (stage, kind) in kinds)
        {
            int stride = kind.First ? family.Strides[stage] : 1;
            string entry = $"{kind.Kind}:{stride}:{family.Activations[stage]}";
            blocks.Add(kind.Kind == "cn" && family.ConvSkip && stage == 0 ? entry + ":skip" : entry);
        }

        string a = Architecture + ".";
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "image_size"] = processor.CropWidth,
            [a + "blocks"] = string.Join(',', blocks),
            [a + "same_padding"] = family.Same,
            [a + "stem_activation"] = family.StemActivation,
            [a + "se_activation"] = family.SeActivation,
            [a + "se_gate"] = family.SeGate,
            [a + "head_activation"] = family.HeadActivation,
            [a + "head_pool_first"] = family.PoolFirst,
            [CanonicalKeys.ImageProcessor] = processor.ToJson().ToJsonString(),
        };
        if (c.Get("label_names") is { ValueKind: JsonValueKind.Array } names)
        {
            md[CanonicalKeys.ClassifierLabels] = names.EnumerateArray().Select(v => v.GetString() ?? "").ToArray();
        }

        return md;
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        var family = FamilyOf(c.GetString("architecture")!)!;
        var flat = FlatIndex(tensors.Keys);
        var collector = new HfWeightCollector();
        foreach (var (rawName, tensor) in tensors)
        {
            if (rawName.EndsWith("num_batches_tracked", StringComparison.Ordinal) || BatchNormTensor().IsMatch(rawName))
            {
                continue;
            }

            string? canonical = null;
            string? bn = null;
            if (rawName == "conv_stem.weight")
            {
                (canonical, bn) = ("stem", "bn1");
            }
            else if (rawName == "conv_head.weight")
            {
                (canonical, bn) = ("head.conv", tensors.ContainsKey("bn2.weight") ? "bn2" : null);
            }
            else if (rawName is "conv_head.bias")
            {
                collector.Add(HfTransforms.ToF32(tensor.ToWeightInfo(), "head.conv.bias"));
                continue;
            }
            else if (rawName.StartsWith("classifier.", StringComparison.Ordinal))
            {
                collector.Add(HfTransforms.ToF32(tensor.ToWeightInfo(), rawName));
                continue;
            }
            else if (Block().Match(rawName) is { Success: true } m)
            {
                string prefix = $"blocks.{m.Groups[1].Value}.{m.Groups[2].Value}.";
                string part = m.Groups[3].Value;
                string leaf = m.Groups[4].Value;
                string index = $"blocks.{flat[(int.Parse(m.Groups[1].Value, CultureInfo.InvariantCulture), int.Parse(m.Groups[2].Value, CultureInfo.InvariantCulture))]}.";
                if (part.StartsWith("se.", StringComparison.Ordinal))
                {
                    var info = tensor.ToWeightInfo();
                    var se = HfTransforms.ToF32(info, index + "se." + (part.EndsWith("reduce", StringComparison.Ordinal) ? "reduce" : "expand") + "." + leaf,
                        leaf == "weight" ? new Shape(info.Shape[0].Fixed, info.Shape[1].Fixed) : null);
                    collector.Add(se);
                    continue;
                }

                bool inverted = tensors.ContainsKey(prefix + "conv_pwl.weight") && tensors.ContainsKey(prefix + "conv_pw.weight");
                (string, string)? target = part switch
                {
                    "conv_pw" when inverted => ("exp", "bn1"),
                    "conv_dw" when inverted => ("dw", "bn2"),
                    "conv_pwl" when inverted => ("proj", "bn3"),
                    "conv_dw" => ("dw", "bn1"),
                    "conv_pw" => ("proj", "bn2"),
                    "conv_exp" => ("exp", "bn1"),
                    "conv_pwl" => ("proj", "bn2"),
                    "conv" => ("exp", "bn1"),
                    _ => null,
                };
                if (target is { } t && leaf == "weight")
                {
                    canonical = index + t.Item1;
                    bn = prefix + t.Item2;
                }
            }

            if (canonical is null)
            {
                collector.AddUnmapped(rawName);
                continue;
            }

            if (bn is null)
            {
                collector.Add(HfTransforms.ToF32(tensor.ToWeightInfo(), canonical + ".weight"));
                continue;
            }

            SafetensorsTensor T(string suffix) => tensors[bn + suffix];
            var (w, b) = HfBatchNormFolding.Fold(tensor.ToWeightInfo(), T(".weight").ToWeightInfo(), T(".bias").ToWeightInfo(),
                T(".running_mean").ToWeightInfo(), T(".running_var").ToWeightInfo(), family.Eps, canonical);
            collector.Add(w);
            collector.Add(b);
        }

        collector.ThrowIfUnmapped(c.ModelType);
        return collector.Weights;
    }

    private static Dictionary<(int Stage, int Block), int> FlatIndex(IEnumerable<string> names)
    {
        var keys = names.Select(n => Block().Match(n)).Where(m => m.Success)
            .Select(m => (Stage: int.Parse(m.Groups[1].Value, CultureInfo.InvariantCulture), Block: int.Parse(m.Groups[2].Value, CultureInfo.InvariantCulture)))
            .Distinct().OrderBy(k => k.Stage).ThenBy(k => k.Block).ToList();
        return keys.Select((k, i) => (k, i)).ToDictionary(p => p.k, p => p.i);
    }

    private static List<(int Stage, (string Kind, bool First) Kind)> BlockKinds(string directory)
    {
        using var header = SafetensorsFile.Open(Path.Combine(directory, "model.safetensors"));
        var names = header.Tensors.Keys.ToHashSet(StringComparer.Ordinal);
        var result = new List<(int, (string, bool))>();
        foreach (var (stage, block) in FlatIndex(names).Keys.OrderBy(k => k.Stage).ThenBy(k => k.Block))
        {
            string p = $"blocks.{stage}.{block}.";
            string kind = names.Contains(p + "conv_pwl.weight") && names.Contains(p + "conv_pw.weight") ? "ir"
                : names.Contains(p + "conv_dw.weight") ? "ds"
                : names.Contains(p + "conv_exp.weight") ? "er"
                : "cn";
            result.Add((stage, (kind, block == 0)));
        }

        return result;
    }

    private static ImageProcessorConfig TimmProcessor(HfConfig c)
    {
        int size = 224;
        double cropPct = 0.875;
        var filter = ImageResampleFilter.Bicubic;
        float[] mean = [0.485f, 0.456f, 0.406f];
        float[] std = [0.229f, 0.224f, 0.225f];
        if (c.Get("pretrained_cfg") is { ValueKind: JsonValueKind.Object } p)
        {
            size = p.TryGetProperty("input_size", out var inputSize) && inputSize.GetArrayLength() == 3 ? inputSize[1].GetInt32() : size;
            cropPct = p.TryGetProperty("crop_pct", out var crop) ? crop.GetDouble() : cropPct;
            filter = p.TryGetProperty("interpolation", out var interpolation) && interpolation.GetString() == "bilinear" ? ImageResampleFilter.Bilinear : ImageResampleFilter.Bicubic;
            mean = p.TryGetProperty("mean", out var m) ? [.. m.EnumerateArray().Select(v => v.GetSingle())] : mean;
            std = p.TryGetProperty("std", out var s) ? [.. s.EnumerateArray().Select(v => v.GetSingle())] : std;
        }

        return new ImageProcessorConfig
        {
            ResizeMode = ImageResizeMode.ShortestEdge,
            ShortestEdge = (int)Math.Floor(size / cropPct),
            CropWidth = size,
            CropHeight = size,
            CropOffsetNearest = true,
            Filter = filter,
            Mean = mean,
            Std = std,
        };
    }

    [GeneratedRegex(@"(^|\.)bn\d\.")]
    private static partial Regex BatchNormTensor();

    [GeneratedRegex(@"^blocks\.(\d+)\.(\d+)\.(conv_dw|conv_pw|conv_pwl|conv_exp|conv|se\.conv_reduce|se\.conv_expand)\.(weight|bias)$")]
    private static partial Regex Block();
}
