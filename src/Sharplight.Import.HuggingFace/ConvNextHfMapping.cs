using System.Globalization;
using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Tensors;
using Sharplight.Vision;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// ConvNeXt V1 checkpoints → canonical <c>convnext</c> (documented on <c>ConvNextClassifierAdapter</c>): Hugging Face <c>convnext</c> (<c>facebook/convnext-tiny-224</c> and fine-tunes; the processor follows
/// <c>preprocessor_config.json</c>) and timm (<c>timm/convnext_tiny.fb_in1k</c> and the other sizes, reported by <see cref="HfConfig.ModelType"/> as <c>timm_convnext</c>; the processor follows
/// <c>pretrained_cfg</c> with timm's rounded crop offsets). ConvNeXt V2 (global response normalization) is not mapped. Convolutional MLPs (timm's atto to nano) are read as linear layers.
/// </summary>
internal sealed partial class ConvNextHfMapping : IHfArchitectureMapping
{
    private static readonly Dictionary<string, (int[] Depths, int[] Dims)> s_timm = new(StringComparer.Ordinal)
    {
        ["convnext_atto"] = ([2, 2, 6, 2], [40, 80, 160, 320]),
        ["convnext_femto"] = ([2, 2, 6, 2], [48, 96, 192, 384]),
        ["convnext_pico"] = ([2, 2, 6, 2], [64, 128, 256, 512]),
        ["convnext_nano"] = ([2, 2, 8, 2], [80, 160, 320, 640]),
        ["convnext_tiny"] = ([3, 3, 9, 3], [96, 192, 384, 768]),
        ["convnext_small"] = ([3, 3, 27, 3], [96, 192, 384, 768]),
        ["convnext_base"] = ([3, 3, 27, 3], [128, 256, 512, 1024]),
        ["convnext_large"] = ([3, 3, 27, 3], [192, 384, 768, 1536]),
        ["convnext_xlarge"] = ([3, 3, 27, 3], [256, 512, 1024, 2048]),
    };

    /// <inheritdoc/>
    public string Architecture => "convnext";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["convnext", HfConfig.TimmConvNextModelType];

    /// <inheritdoc/>
    public bool NeedsTokenizer => false;

    private static bool IsTimm(HfConfig c) => c.ModelType == HfConfig.TimmConvNextModelType;

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig c)
    {
        if (IsTimm(c))
        {
            string name = c.GetString("architecture") ?? "";
            if (!s_timm.ContainsKey(name))
            {
                yield return $"timm: the ConvNeXt variant '{name}' is not supported (atto, femto, pico, nano, tiny, small, base, large, xlarge; V1 only).";
            }
        }
        else if (c.GetString("hidden_act") is { } act && act != "gelu")
        {
            yield return $"convnext: the activation '{act}' is not supported.";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        int[] depths;
        int[] dims;
        ImageProcessorConfig processor;
        if (IsTimm(c))
        {
            (depths, dims) = s_timm[c.GetString("architecture")!];
            processor = TimmProcessor(c);
        }
        else
        {
            depths = [.. c.Get("depths")!.Value.EnumerateArray().Select(v => v.GetInt32())];
            dims = [.. c.Get("hidden_sizes")!.Value.EnumerateArray().Select(v => v.GetInt32())];
            processor = HfProcessor(c.Directory);
        }

        string a = Architecture + ".";
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "image_size"] = HfImageProcessorConfig.InputSize(processor, 224),
            [a + "depths"] = string.Join(',', depths),
            [a + "hidden_sizes"] = string.Join(',', dims),
            [a + "layer_norm_eps"] = IsTimm(c) ? 1e-6f : (float)(c.GetDouble("layer_norm_eps") ?? 1e-12),
            [CanonicalKeys.ImageProcessor] = processor.ToJson().ToJsonString(),
        };
        if (c.Get("id2label") is { ValueKind: JsonValueKind.Object } map)
        {
            md[CanonicalKeys.ClassifierLabels] = map.EnumerateObject()
                .Select(p => (Index: int.Parse(p.Name, CultureInfo.InvariantCulture), Label: p.Value.GetString() ?? ""))
                .OrderBy(p => p.Index).Select(p => p.Label).ToArray();
        }
        else if (c.Get("label_names") is { ValueKind: JsonValueKind.Array } names)
        {
            md[CanonicalKeys.ClassifierLabels] = names.EnumerateArray().Select(v => v.GetString() ?? "").ToArray();
        }

        return md;
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        var collector = new HfWeightCollector();
        bool timm = IsTimm(c);
        foreach (var (rawName, tensor) in tensors)
        {
            string? canonical = timm ? TimmName(rawName) : HfName(rawName);
            if (canonical is null)
            {
                collector.AddUnmapped(rawName);
                continue;
            }

            var info = tensor.ToWeightInfo();
            bool linear = canonical.Contains(".pw", StringComparison.Ordinal) && canonical.EndsWith(".weight", StringComparison.Ordinal);
            collector.Add(linear && info.Shape.Rank == 4
                ? HfTransforms.ToF32(info, canonical, new Shape(info.Shape[0].Fixed, info.Shape[1].Fixed))
                : HfTransforms.ToF32(info, canonical));
        }

        collector.ThrowIfUnmapped(c.ModelType);
        return collector.Weights;
    }

    private static string? HfName(string n)
    {
        if (n is "classifier.weight" or "classifier.bias")
        {
            return n;
        }

        if (n.StartsWith("convnext.layernorm.", StringComparison.Ordinal))
        {
            return "head.norm." + n[^4..].TrimStart('.');
        }

        if (n.StartsWith("convnext.embeddings.patch_embeddings.", StringComparison.Ordinal))
        {
            return "stem.conv." + n.Split('.')[^1];
        }

        if (n.StartsWith("convnext.embeddings.layernorm.", StringComparison.Ordinal))
        {
            return "stem.norm." + n.Split('.')[^1];
        }

        if (HfDown().Match(n) is { Success: true } d)
        {
            return $"stages.{d.Groups[1].Value}.down." + (d.Groups[2].Value == "0" ? "norm." : "conv.") + d.Groups[3].Value;
        }

        if (HfLayer().Match(n) is { Success: true } l)
        {
            string kind = l.Groups[3].Value switch
            {
                "dwconv" => "dw",
                "layernorm" => "norm",
                "pwconv1" => "pw1",
                "pwconv2" => "pw2",
                _ => "gamma",
            };
            return $"stages.{l.Groups[1].Value}.blocks.{l.Groups[2].Value}.{kind}." + (kind == "gamma" ? "weight" : l.Groups[4].Value);
        }

        return null;
    }

    private static string? TimmName(string n)
    {
        if (n is "head.fc.weight" or "head.fc.bias")
        {
            return "classifier." + n[^4..].TrimStart('.');
        }

        if (n.StartsWith("head.norm.", StringComparison.Ordinal))
        {
            return n;
        }

        if (n.StartsWith("stem.0.", StringComparison.Ordinal))
        {
            return "stem.conv." + n[7..];
        }

        if (n.StartsWith("stem.1.", StringComparison.Ordinal))
        {
            return "stem.norm." + n[7..];
        }

        if (TimmDown().Match(n) is { Success: true } d)
        {
            return $"stages.{d.Groups[1].Value}.down." + (d.Groups[2].Value == "0" ? "norm." : "conv.") + d.Groups[3].Value;
        }

        if (TimmBlock().Match(n) is { Success: true } l)
        {
            string kind = l.Groups[3].Value switch
            {
                "conv_dw" => "dw",
                "norm" => "norm",
                "mlp.fc1" => "pw1",
                "mlp.fc2" => "pw2",
                _ => "gamma",
            };
            return $"stages.{l.Groups[1].Value}.blocks.{l.Groups[2].Value}.{kind}." + (kind == "gamma" ? "weight" : l.Groups[4].Value);
        }

        return null;
    }

    private static ImageProcessorConfig HfProcessor(string directory)
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

        return new ImageProcessorConfig { ResizeMode = ImageResizeMode.ShortestEdge, ShortestEdge = (int)(size / cropPct), CropWidth = size, CropHeight = size, Filter = filter, Mean = mean, Std = std };
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

    [GeneratedRegex(@"^convnext\.encoder\.stages\.(\d+)\.downsampling_layer\.(0|1)\.(weight|bias)$")]
    private static partial Regex HfDown();

    [GeneratedRegex(@"^convnext\.encoder\.stages\.(\d+)\.layers\.(\d+)\.(dwconv|layernorm|pwconv1|pwconv2|layer_scale_parameter)(?:\.(weight|bias))?$")]
    private static partial Regex HfLayer();

    [GeneratedRegex(@"^stages\.(\d+)\.downsample\.(0|1)\.(weight|bias)$")]
    private static partial Regex TimmDown();

    [GeneratedRegex(@"^stages\.(\d+)\.blocks\.(\d+)\.(conv_dw|norm|mlp\.fc1|mlp\.fc2|gamma)(?:\.(weight|bias))?$")]
    private static partial Regex TimmBlock();
}
