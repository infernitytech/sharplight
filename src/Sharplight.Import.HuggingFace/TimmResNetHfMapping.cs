using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Vision;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// timm ResNet checkpoints (<c>timm/resnet50.a1_in1k</c> and the like: a <c>config.json</c> with <c>architecture</c> and <c>pretrained_cfg</c> and no <c>model_type</c>,
/// reported by <see cref="HfConfig.ModelType"/> as <c>timm_resnet</c>) → canonical <c>resnet</c> (documented on <c>ResNetClassifierAdapter</c>). Batch norms are folded into the
/// convolutions. The processor follows <c>pretrained_cfg</c>: the shortest edge resized to <c>size / crop_pct</c>, a center crop of <c>size</c>, mean and std. Plain ResNets
/// (basic or bottleneck blocks, stride in the 3x3 convolution, 7x7 stem) and wide ResNets are mapped; the deep-stem, ResNeXt, squeeze-excitation and pre-activation
/// variants are rejected with a reason.
/// </summary>
internal sealed partial class TimmResNetHfMapping : IHfArchitectureMapping
{
    private static readonly Dictionary<string, (int[] Depths, bool Bottleneck)> s_architectures = new(StringComparer.Ordinal)
    {
        ["resnet18"] = ([2, 2, 2, 2], false),
        ["resnet26"] = ([2, 2, 2, 2], true),
        ["resnet34"] = ([3, 4, 6, 3], false),
        ["resnet50"] = ([3, 4, 6, 3], true),
        ["resnet101"] = ([3, 4, 23, 3], true),
        ["resnet152"] = ([3, 8, 36, 3], true),
        ["resnet200"] = ([3, 24, 36, 3], true),
        ["wide_resnet50_2"] = ([3, 4, 6, 3], true),
        ["wide_resnet101_2"] = ([3, 4, 23, 3], true),
    };

    /// <inheritdoc/>
    public string Architecture => "resnet";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = [HfConfig.TimmResNetModelType];

    /// <inheritdoc/>
    public bool NeedsTokenizer => false;

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig c)
    {
        string name = c.GetString("architecture") ?? "";
        if (!s_architectures.ContainsKey(name))
        {
            yield return $"timm: the ResNet variant '{name}' is not supported (resnet18, 26, 34, 50, 101, 152, 200, wide_resnet50_2, wide_resnet101_2).";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        var (depths, bottleneck) = s_architectures[c.GetString("architecture")!];
        var cfg = c.Get("pretrained_cfg");
        int size = 224;
        double cropPct = 0.875;
        var filter = ImageResampleFilter.Bicubic;
        float[] mean = [0.485f, 0.456f, 0.406f];
        float[] std = [0.229f, 0.224f, 0.225f];
        if (cfg is { ValueKind: JsonValueKind.Object } p)
        {
            size = p.TryGetProperty("input_size", out var inputSize) && inputSize.GetArrayLength() == 3 ? inputSize[1].GetInt32() : size;
            cropPct = p.TryGetProperty("crop_pct", out var crop) ? crop.GetDouble() : cropPct;
            filter = p.TryGetProperty("interpolation", out var interpolation) && interpolation.GetString() == "bilinear" ? ImageResampleFilter.Bilinear : ImageResampleFilter.Bicubic;
            mean = p.TryGetProperty("mean", out var m) ? [.. m.EnumerateArray().Select(v => v.GetSingle())] : mean;
            std = p.TryGetProperty("std", out var s) ? [.. s.EnumerateArray().Select(v => v.GetSingle())] : std;
        }

        var processor = new ImageProcessorConfig
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
        string a = Architecture + ".";
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "image_size"] = size,
            [a + "backbone.depths"] = string.Join(',', depths),
            [a + "backbone.bottleneck"] = bottleneck,
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
        var collector = new HfWeightCollector();
        foreach (var (rawName, tensor) in tensors)
        {
            if (rawName.EndsWith("num_batches_tracked", StringComparison.Ordinal) || BatchNormTensor().IsMatch(rawName))
            {
                continue;
            }

            if (rawName is "fc.weight" or "fc.bias")
            {
                collector.Add(HfTransforms.ToF32(tensor.ToWeightInfo(), "classifier." + rawName[3..]));
                continue;
            }

            string bnName;
            string canonical;
            if (rawName == "conv1.weight")
            {
                bnName = "bn1";
                canonical = "backbone.stem.0";
            }
            else if (BlockConv().Match(rawName) is { Success: true } conv)
            {
                string stage = (int.Parse(conv.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture) - 1).ToString(System.Globalization.CultureInfo.InvariantCulture);
                bool shortcut = conv.Groups[3].Value == "downsample.0";
                bnName = $"layer{conv.Groups[1].Value}.{conv.Groups[2].Value}." + (shortcut ? "downsample.1" : "bn" + conv.Groups[3].Value[^1]);
                canonical = $"backbone.stages.{stage}.{conv.Groups[2].Value}." + (shortcut ? "short" : conv.Groups[3].Value);
            }
            else
            {
                collector.AddUnmapped(rawName);
                continue;
            }

            SafetensorsTensor T(string suffix) => tensors[bnName + suffix];
            var (w, b) = HfBatchNormFolding.Fold(tensor.ToWeightInfo(), T(".weight").ToWeightInfo(), T(".bias").ToWeightInfo(),
                T(".running_mean").ToWeightInfo(), T(".running_var").ToWeightInfo(), 1e-5f, canonical);
            collector.Add(w);
            collector.Add(b);
        }

        collector.ThrowIfUnmapped(c.ModelType);
        return collector.Weights;
    }

    [GeneratedRegex(@"^layer(\d)\.(\d+)\.(conv\d|downsample\.0)\.weight$")]
    private static partial Regex BlockConv();

    [GeneratedRegex(@"(^|\.)(bn\d|downsample\.1)\.")]
    private static partial Regex BatchNormTensor();
}
