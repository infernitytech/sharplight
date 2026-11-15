using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Hugging Face Depth Anything (<c>depth_anything</c>: DepthAnythingForDepthEstimation on a DINOv2 backbone) → canonical <c>dpt_depth</c> (documented on
/// <c>DepthEstimationAdapter</c>). The DINOv2 backbone is mapped by <see cref="VitHfMapping"/> from the checkpoint's <c>backbone_config</c> (with
/// Dinov2Config's defaults for the keys it leaves out) and stored under <c>backbone.</c>. The graph runs on the square input of the stored processor
/// (518 px, no aspect-ratio preservation). Only relative depth (ReLU head) is supported.
/// </summary>
internal sealed partial class DepthAnythingHfMapping : IHfArchitectureMapping
{
    private readonly VitHfMapping _backbone = new();

    /// <inheritdoc/>
    public string Architecture => "dpt_depth";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["depth_anything"];

    /// <inheritdoc/>
    public bool NeedsTokenizer => false;

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig c)
    {
        if ((c.GetString("depth_estimation_type") ?? "relative") != "relative")
        {
            yield return "depth_anything: only relative depth estimation is supported.";
        }

        if (c.Get("backbone_config") is not { ValueKind: JsonValueKind.Object })
        {
            yield return "depth_anything: the checkpoint has no backbone_config.";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        string a = Architecture + ".";
        var md = new Dictionary<string, object>(StringComparer.Ordinal);
        using var backbone = BackboneConfig(c);
        foreach (var (key, value) in _backbone.MapHyperParameters(backbone, options))
        {
            md[key.StartsWith("vit.", StringComparison.Ordinal) ? a + "backbone." + key["vit.".Length..] : key] = value;
        }

        var outputs = c.Get("backbone_config")!.Value.GetProperty("out_indices").EnumerateArray().Select(v => v.GetInt32());
        md[a + "tap_layers"] = string.Join(',', outputs);
        md[a + "neck.hidden_sizes"] = string.Join(',', c.Get("neck_hidden_sizes")!.Value.EnumerateArray().Select(v => v.GetInt32()));
        md[a + "neck.factors"] = string.Join(',', c.Get("reassemble_factors")!.Value.EnumerateArray().Select(v => v.GetDouble().ToString("R", CultureInfo.InvariantCulture)));
        md[a + "fusion_hidden_size"] = c.Int("fusion_hidden_size");
        md[a + "head.hidden_size"] = c.Int("head_hidden_size");
        return md;
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        using var backbone = BackboneConfig(c);
        var inner = tensors.Where(t => t.Key.StartsWith("backbone.", StringComparison.Ordinal)).ToDictionary(t => t.Key["backbone.".Length..], t => t.Value, StringComparer.Ordinal);
        var collector = new HfWeightCollector();
        foreach (var (name, weight) in _backbone.MapWeights(backbone, inner, options))
        {
            collector.Add(RenameWeight(weight, "backbone." + name));
        }

        foreach (var (rawName, tensor) in tensors)
        {
            if (rawName.StartsWith("backbone.", StringComparison.Ordinal))
            {
                continue;
            }

            var info = tensor.ToWeightInfo();
            string? canonical = Rename(rawName);
            if (canonical is null)
            {
                collector.AddUnmapped(rawName);
                continue;
            }

            collector.Add(info.As(canonical));
        }

        collector.ThrowIfUnmapped(c.ModelType);
        return collector.Weights;
    }

    private static WeightInfo RenameWeight(WeightInfo weight, string name) => weight.As(name);

    private static string? Rename(string name)
    {
        if (Regex.Match(name, @"^neck\.reassemble_stage\.layers\.(\d+)\.(projection|resize)\.(weight|bias)$") is { Success: true } reassemble)
        {
            return $"neck.reassemble.{reassemble.Groups[1].Value}.{(reassemble.Groups[2].Value == "projection" ? "proj" : "resize")}.{reassemble.Groups[3].Value}";
        }

        if (Regex.Match(name, @"^neck\.convs\.(\d+)\.weight$") is { Success: true } conv)
        {
            return $"neck.convs.{conv.Groups[1].Value}.weight";
        }

        if (Regex.Match(name, @"^neck\.fusion_stage\.layers\.(\d+)\.(projection|residual_layer([12])\.convolution([12]))\.(weight|bias)$") is { Success: true } fusion)
        {
            string module = fusion.Groups[2].Value == "projection" ? "proj" : $"res{fusion.Groups[3].Value}.conv{fusion.Groups[4].Value}";
            return $"neck.fusion.{fusion.Groups[1].Value}.{module}.{fusion.Groups[5].Value}";
        }

        return name.StartsWith("head.conv", StringComparison.Ordinal) ? name : null;
    }

    // The backbone's configuration as a standalone DINOv2 configuration (Dinov2Config defaults for the keys the nested one leaves out).
    private static HfConfig BackboneConfig(HfConfig c)
    {
        var nested = JsonNode.Parse(c.Get("backbone_config")!.Value.GetRawText())!.AsObject();
        nested["model_type"] = "dinov2";
        foreach (var (key, value) in new (string, JsonNode)[] { ("num_hidden_layers", 12), ("layer_norm_eps", 1e-6), ("hidden_act", "gelu"), ("mlp_ratio", 4), ("patch_size", 14) })
        {
            nested.TryAdd(key, value);
        }

        return HfConfig.Parse(nested.ToJsonString(), c.Directory);
    }
}
