using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Imaging;
using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Hugging Face DETR (<c>detr</c>: DetrForObjectDetection with the timm ResNet backbone) → canonical <c>detr</c> (documented on <c>DetrAdapter</c>).
/// Frozen batch norms are folded into the backbone convolutions. The graph runs on a fixed square canvas (<see cref="InputSize"/>): the stored
/// processor resizes every image to it (the checkpoint's own processor resizes to 800/1333 pixels preserving the aspect ratio, which a static graph
/// cannot follow).
/// </summary>
internal sealed partial class DetrHfMapping : IHfArchitectureMapping
{
    /// <summary>The side of the square model input in pixels.</summary>
    public const int InputSize = 640;

    private const string BackbonePrefix = "model.backbone.conv_encoder.model.";

    private static readonly Dictionary<string, (int[] Depths, bool Bottleneck)> s_backbones = new(StringComparer.Ordinal)
    {
        ["resnet18"] = ([2, 2, 2, 2], false),
        ["resnet34"] = ([3, 4, 6, 3], false),
        ["resnet50"] = ([3, 4, 6, 3], true),
        ["resnet101"] = ([3, 4, 23, 3], true),
    };

    /// <inheritdoc/>
    public string Architecture => "detr";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["detr"];

    /// <inheritdoc/>
    public bool NeedsTokenizer => false;

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig c)
    {
        if (!s_backbones.ContainsKey(c.GetString("backbone") ?? "resnet50"))
        {
            yield return $"detr: backbone '{c.GetString("backbone")}' is not supported (resnet18, 34, 50, 101).";
        }

        if (c.GetBool("dilation") == true)
        {
            yield return "detr: a dilated backbone is not supported.";
        }

        if ((c.GetString("position_embedding_type") ?? "sine") != "sine")
        {
            yield return "detr: only sine position embeddings are supported.";
        }

        if (c.GetBool("use_timm_backbone") == false)
        {
            yield return "detr: only checkpoints with the timm backbone are supported.";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        string a = Architecture + ".";
        var (depths, bottleneck) = s_backbones[c.GetString("backbone") ?? "resnet50"];
        var processor = HfImageProcessorConfig.Read(c.Directory, InputSize) with { ResizeMode = ImageResizeMode.Exact, Width = InputSize, Height = InputSize };
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "hidden_size"] = c.Int("d_model"),
            [a + "attention.head_count"] = c.Int("decoder_attention_heads"),
            [a + "encoder.layer_count"] = c.Int("encoder_layers"),
            [a + "decoder.layer_count"] = c.Int("decoder_layers"),
            [a + "ffn_hidden_size"] = c.Int("decoder_ffn_dim"),
            [a + "mlp.activation"] = c.GetString("activation_function") ?? "relu",
            [a + "query_count"] = c.Int("num_queries"),
            [a + "image_size"] = InputSize,
            [a + "backbone.depths"] = string.Join(',', depths),
            [a + "backbone.bottleneck"] = bottleneck,
            [CanonicalKeys.ImageProcessor] = processor.ToJson().ToJsonString(),
        };
        if (Labels(c) is { Length: > 0 } labels)
        {
            md[CanonicalKeys.ClassifierLabels] = labels;
        }

        return md;
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        var collector = new HfWeightCollector();
        foreach (var (rawName, tensor) in tensors)
        {
            string name = rawName;
            var info = tensor.ToWeightInfo();
            if (name.StartsWith(BackbonePrefix, StringComparison.Ordinal))
            {
                MapBackbone(name[BackbonePrefix.Length..], tensors, collector, rawName);
                continue;
            }

            if (name.EndsWith("num_batches_tracked", StringComparison.Ordinal))
            {
                continue;
            }

            string? canonical = Rename(name);
            if (canonical is null)
            {
                collector.AddUnmapped(rawName);
                continue;
            }

            collector.Add(info.Shape.Rank == 1 || canonical.StartsWith("input_proj.", StringComparison.Ordinal) || canonical == "query_position.weight"
                ? HfTransforms.ToF32(info, canonical)
                : info.As(canonical));
        }

        collector.ThrowIfUnmapped(c.ModelType);
        return collector.Weights;
    }

    private static string? Rename(string name)
    {
        if (name.StartsWith("model.input_projection.", StringComparison.Ordinal))
        {
            return "input_proj." + name["model.input_projection.".Length..];
        }

        if (name == "model.query_position_embeddings.weight")
        {
            return "query_position.weight";
        }

        if (name.StartsWith("model.decoder.layernorm.", StringComparison.Ordinal))
        {
            return "decoder.norm." + name["model.decoder.layernorm.".Length..];
        }

        if (name.StartsWith("class_labels_classifier.", StringComparison.Ordinal))
        {
            return "class_head." + name["class_labels_classifier.".Length..];
        }

        if (BoxHead().Match(name) is { Success: true } box)
        {
            return $"box_head.{box.Groups[1].Value}.{box.Groups[2].Value}";
        }

        var m = Layer().Match(name);
        if (!m.Success || !s_layerModules.TryGetValue(m.Groups[3].Value, out var module))
        {
            return null;
        }

        return $"{m.Groups[1].Value}.layers.{m.Groups[2].Value}.{module}.{m.Groups[4].Value}";
    }

    private static void MapBackbone(string name, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HfWeightCollector collector, string rawName)
    {
        if (name.EndsWith("num_batches_tracked", StringComparison.Ordinal) || BatchNormTensor().IsMatch(name))
        {
            return;
        }

        string bnName;
        string canonical;
        if (name == "conv1.weight")
        {
            bnName = "bn1";
            canonical = "backbone.stem.0";
        }
        else if (BlockConv().Match(name) is { Success: true } conv)
        {
            string stage = (int.Parse(conv.Groups[1].Value) - 1).ToString(System.Globalization.CultureInfo.InvariantCulture);
            string prefix = $"layer{conv.Groups[1].Value}.{conv.Groups[2].Value}.";
            bool shortcut = conv.Groups[3].Value == "downsample.0";
            bnName = prefix + (shortcut ? "downsample.1" : "bn" + conv.Groups[3].Value[^1]);
            canonical = $"backbone.stages.{stage}.{conv.Groups[2].Value}." + (shortcut ? "short" : conv.Groups[3].Value);
        }
        else
        {
            collector.AddUnmapped(rawName);
            return;
        }

        SafetensorsTensor T(string suffix) => tensors[BackbonePrefix + bnName + suffix];
        var (w, b) = HfBatchNormFolding.Fold(tensors[rawName].ToWeightInfo(), T(".weight").ToWeightInfo(), T(".bias").ToWeightInfo(),
            T(".running_mean").ToWeightInfo(), T(".running_var").ToWeightInfo(), 1e-5f, canonical);
        collector.Add(w);
        collector.Add(b);
    }

    private static string[] Labels(HfConfig c)
    {
        if (c.Get("id2label") is not { ValueKind: JsonValueKind.Object } map)
        {
            return [];
        }

        return [.. map.EnumerateObject().Select(p => (Index: int.Parse(p.Name, System.Globalization.CultureInfo.InvariantCulture), Label: p.Value.GetString() ?? "")).OrderBy(p => p.Index).Select(p => p.Label)];
    }

    [GeneratedRegex(@"^bbox_predictor\.layers\.(\d+)\.(weight|bias)$")]
    private static partial Regex BoxHead();

    [GeneratedRegex(@"^model\.(encoder|decoder)\.layers\.(\d+)\.(.+)\.(weight|bias)$")]
    private static partial Regex Layer();

    [GeneratedRegex(@"^layer(\d)\.(\d+)\.(conv\d|downsample\.0)\.weight$")]
    private static partial Regex BlockConv();

    [GeneratedRegex(@"(^|\.)(bn\d|downsample\.1)\.")]
    private static partial Regex BatchNormTensor();

    private static readonly Dictionary<string, string> s_layerModules = new(StringComparer.Ordinal)
    {
        ["self_attn.q_proj"] = "self_attn.q",
        ["self_attn.k_proj"] = "self_attn.k",
        ["self_attn.v_proj"] = "self_attn.v",
        ["self_attn.out_proj"] = "self_attn.o",
        ["encoder_attn.q_proj"] = "cross_attn.q",
        ["encoder_attn.k_proj"] = "cross_attn.k",
        ["encoder_attn.v_proj"] = "cross_attn.v",
        ["encoder_attn.out_proj"] = "cross_attn.o",
        ["self_attn_layer_norm"] = "self_attn_norm",
        ["encoder_attn_layer_norm"] = "cross_attn_norm",
        ["final_layer_norm"] = "ffn_norm",
        ["fc1"] = "ffn.up",
        ["fc2"] = "ffn.down",
    };
}
