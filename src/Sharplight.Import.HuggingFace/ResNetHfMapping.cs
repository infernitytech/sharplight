using System.Globalization;
using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Vision;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Hugging Face <c>resnet</c> checkpoints (<c>microsoft/resnet-50</c>, <c>resnet-18</c> and their fine-tunes: <c>ResNetForImageClassification</c> with an embedder, stages of basic or bottleneck
/// layers with convolution + batch norm pairs) → canonical <c>resnet</c> (documented on <c>ResNetClassifierAdapter</c>). Batch norms are folded into the convolutions; the
/// processor follows <c>preprocessor_config.json</c> (ConvNeXt style: shortest edge <c>size / crop_pct</c> and a center crop below 384 px, a warped square from 384 px up). Checkpoints with a
/// different activation, a stride in the first 1x1 convolution (<c>downsample_in_bottleneck</c>) or a stride in the first stage are rejected.
/// </summary>
internal sealed partial class ResNetHfMapping : IHfArchitectureMapping
{
    /// <inheritdoc/>
    public string Architecture => "resnet";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["resnet"];

    /// <inheritdoc/>
    public bool NeedsTokenizer => false;

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig c)
    {
        if (c.GetString("hidden_act") is { } act && act != "relu")
        {
            yield return $"resnet: the activation '{act}' is not supported (relu only).";
        }

        if (c.GetBool("downsample_in_bottleneck") == true)
        {
            yield return "resnet: downsample_in_bottleneck (stride in the first 1x1 convolution) is not supported.";
        }

        if (c.GetBool("downsample_in_first_stage") == true)
        {
            yield return "resnet: downsample_in_first_stage is not supported.";
        }

        if ((c.GetInt("embedding_size") ?? 64) != 64)
        {
            yield return "resnet: embedding sizes other than 64 are not supported.";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        var depths = c.Get("depths")!.Value.EnumerateArray().Select(v => v.GetInt32()).ToArray();
        var processor = Processor(c.Directory);
        string a = Architecture + ".";
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "image_size"] = HfImageProcessorConfig.InputSize(processor, 224),
            [a + "backbone.depths"] = string.Join(',', depths),
            [a + "backbone.bottleneck"] = (c.GetString("layer_type") ?? "bottleneck") == "bottleneck",
            [CanonicalKeys.ImageProcessor] = processor.ToJson().ToJsonString(),
        };
        if (c.Get("id2label") is { ValueKind: JsonValueKind.Object } map)
        {
            md[CanonicalKeys.ClassifierLabels] = map.EnumerateObject()
                .Select(p => (Index: int.Parse(p.Name, CultureInfo.InvariantCulture), Label: p.Value.GetString() ?? ""))
                .OrderBy(p => p.Index).Select(p => p.Label).ToArray();
        }

        return md;
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        var collector = new HfWeightCollector();
        foreach (var (rawName, tensor) in tensors)
        {
            if (rawName.EndsWith("num_batches_tracked", StringComparison.Ordinal) || Normalization().IsMatch(rawName))
            {
                continue;
            }

            if (rawName is "classifier.1.weight" or "classifier.1.bias")
            {
                collector.Add(HfTransforms.ToF32(tensor.ToWeightInfo(), "classifier." + rawName[(rawName.LastIndexOf('.') + 1)..]));
                continue;
            }

            string convBase;
            string canonical;
            if (rawName == "resnet.embedder.embedder.convolution.weight")
            {
                convBase = "resnet.embedder.embedder";
                canonical = "backbone.stem.0";
            }
            else if (LayerConv().Match(rawName) is { Success: true } m)
            {
                convBase = rawName[..^".convolution.weight".Length];
                canonical = $"backbone.stages.{m.Groups[1].Value}.{m.Groups[2].Value}." + (m.Groups[3].Success ? "conv" + (int.Parse(m.Groups[3].Value, CultureInfo.InvariantCulture) + 1).ToString(CultureInfo.InvariantCulture) : "short");
            }
            else
            {
                collector.AddUnmapped(rawName);
                continue;
            }

            SafetensorsTensor T(string suffix) => tensors[convBase + ".normalization" + suffix];
            var (w, b) = HfBatchNormFolding.Fold(tensor.ToWeightInfo(), T(".weight").ToWeightInfo(), T(".bias").ToWeightInfo(),
                T(".running_mean").ToWeightInfo(), T(".running_var").ToWeightInfo(), 1e-5f, canonical);
            collector.Add(w);
            collector.Add(b);
        }

        collector.ThrowIfUnmapped(c.ModelType);
        return collector.Weights;
    }

    private static ImageProcessorConfig Processor(string directory)
    {
        string path = Path.Combine(directory, "preprocessor_config.json");
        using var document = File.Exists(path) ? JsonDocument.Parse(File.ReadAllBytes(path)) : null;
        var root = document?.RootElement ?? default;
        int size = 224;
        if (document is not null && root.TryGetProperty("size", out var s))
        {
            size = s.ValueKind == JsonValueKind.Number ? s.GetInt32() : s.TryGetProperty("shortest_edge", out var edge) ? edge.GetInt32() : s.GetProperty("height").GetInt32();
        }

        double cropPct = document is not null && root.TryGetProperty("crop_pct", out var cp) && cp.ValueKind == JsonValueKind.Number ? cp.GetDouble() : 224.0 / 256.0;
        float[] Floats(string key, float[] fallback) => document is not null && root.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.Array ? [.. v.EnumerateArray().Select(x => x.GetSingle())] : fallback;
        var filter = document is not null && root.TryGetProperty("resample", out var r) && r.ValueKind == JsonValueKind.Number && r.GetInt32() is 2 or 4 or 5 ? ImageResampleFilter.Bilinear : ImageResampleFilter.Bicubic;
        var mean = Floats("image_mean", [0.485f, 0.456f, 0.406f]);
        var std = Floats("image_std", [0.229f, 0.224f, 0.225f]);
        if (size >= 384)
        {
            return new ImageProcessorConfig { ResizeMode = ImageResizeMode.Exact, Width = size, Height = size, Filter = filter, Mean = mean, Std = std };
        }

        return new ImageProcessorConfig
        {
            ResizeMode = ImageResizeMode.ShortestEdge,
            ShortestEdge = (int)(size / cropPct),
            CropWidth = size,
            CropHeight = size,
            Filter = filter,
            Mean = mean,
            Std = std,
        };
    }

    [GeneratedRegex(@"\.normalization\.")]
    private static partial Regex Normalization();

    [GeneratedRegex(@"^resnet\.encoder\.stages\.(\d+)\.layers\.(\d+)\.(?:layer\.(\d+)|shortcut)\.convolution\.weight$")]
    private static partial Regex LayerConv();
}
