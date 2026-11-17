using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Tensors;
using Sharplight.Vision;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Hugging Face YOLOS (<c>yolos</c>: YolosForObjectDetection, a plain vision transformer with detection tokens appended to the patch tokens and DETR's class and box heads) →
/// canonical <c>yolos</c> (documented on <c>YolosAdapter</c>). The graph runs on a fixed square canvas (<see cref="InputSize"/>; the checkpoint's own processor resizes the shortest
/// edge to 512 pixels preserving the aspect ratio, which a static graph cannot follow): the position table (and the mid-layer tables of the larger models) are resampled bicubically
/// to that grid at import, as PyTorch does at run time for the image's grid.
/// </summary>
internal sealed partial class YolosHfMapping : IHfArchitectureMapping
{
    /// <summary>The side of the square model input in pixels.</summary>
    public const int InputSize = 640;

    private static readonly Dictionary<string, string> s_layerModules = new(StringComparer.Ordinal)
    {
        ["layernorm_before"] = "input_norm",
        ["layernorm_after"] = "post_attention_norm",
        ["attention.attention.query"] = "attention.q",
        ["attention.attention.key"] = "attention.k",
        ["attention.attention.value"] = "attention.v",
        ["attention.output.dense"] = "attention.o",
        ["intermediate.dense"] = "mlp.up",
        ["output.dense"] = "mlp.down",
    };

    /// <inheritdoc/>
    public string Architecture => "yolos";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["yolos"];

    /// <inheritdoc/>
    public bool NeedsTokenizer => false;

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig c)
    {
        if (c.Get("image_size") is not { ValueKind: JsonValueKind.Array } size || size.GetArrayLength() != 2)
        {
            yield return "yolos: image_size must be [height, width].";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        string a = Architecture + ".";
        var processor = HfImageProcessorConfig.Read(c.Directory, InputSize) with { ResizeMode = ImageResizeMode.Exact, Width = InputSize, Height = InputSize };
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = c.Int("num_hidden_layers"),
            [a + "hidden_size"] = c.Int("hidden_size"),
            [a + "ffn_hidden_size"] = c.Int("intermediate_size"),
            [a + "attention.head_count"] = c.Int("num_attention_heads"),
            [a + "norm_eps"] = (float)(c.GetDouble("layer_norm_eps") ?? 1e-12),
            [a + "mlp.activation"] = (c.GetString("hidden_act") ?? "gelu") switch
            {
                "gelu" => "gelu",
                "gelu_new" or "gelu_pytorch_tanh" or "gelu_fast" => "gelu_tanh",
                "relu" => "relu",
                var other => throw new NotSupportedException($"yolos: activation '{other}' is not supported."),
            },
            [a + "patch_size"] = c.Int("patch_size"),
            [a + "image_size"] = InputSize,
            [a + "detection_token_count"] = c.GetInt("num_detection_tokens") ?? 100,
            [a + "mid_position_embeddings"] = c.GetBool("use_mid_position_embeddings") == true,
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
        int hidden = c.Int("hidden_size");
        int patch = c.Int("patch_size");
        int detection = c.GetInt("num_detection_tokens") ?? 100;
        var size = c.Get("image_size")!.Value;
        int sourceHeight = size[0].GetInt32() / patch;
        int sourceWidth = size[1].GetInt32() / patch;
        int grid = InputSize / patch;
        var collector = new HfWeightCollector();
        foreach (var (rawName, tensor) in tensors)
        {
            var info = tensor.ToWeightInfo();
            string name = rawName.StartsWith("vit.", StringComparison.Ordinal) ? rawName["vit.".Length..] : rawName;
            switch (name)
            {
                case "embeddings.cls_token":
                    collector.Add(HfTransforms.ToF32(info, "class_embedding", new Shape(hidden)));
                    continue;
                case "embeddings.detection_tokens":
                    collector.Add(HfTransforms.ToF32(info, "detection_embedding.weight", new Shape(detection, hidden)));
                    continue;
                case "embeddings.patch_embeddings.projection.weight":
                    collector.Add(HfTransforms.Plain(info, "patch_embed.weight"));
                    continue;
                case "embeddings.patch_embeddings.projection.bias":
                    collector.Add(HfTransforms.Plain(info, "patch_embed.bias"));
                    continue;
                case "embeddings.position_embeddings":
                    collector.Add(PositionTable(info, "position_embedding.weight", 1, hidden, detection, sourceHeight, sourceWidth, grid));
                    continue;
                case "encoder.mid_position_embeddings":
                    collector.Add(PositionTable(info, "mid_position_embedding.weight", (int)(info.Shape.Elements() / ((1 + (long)sourceHeight * sourceWidth + detection) * hidden)), hidden, detection,
                        sourceHeight, sourceWidth, grid));
                    continue;
                case "layernorm.weight" or "layernorm.bias":
                    collector.Add(HfTransforms.Plain(info, "output_norm." + name["layernorm.".Length..]));
                    continue;
            }

            if (HeadLayer().Match(rawName) is { Success: true } head)
            {
                string canonical = $"{(head.Groups[1].Value == "class_labels_classifier" ? "class_head" : "box_head")}.{head.Groups[2].Value}.{head.Groups[3].Value}";
                collector.Add(HfTransforms.Plain(info, canonical));
                continue;
            }

            if (Layer().Match(name) is { Success: true } layer)
            {
                string rest = layer.Groups[2].Value;
                int dot = rest.LastIndexOf('.');
                if (dot > 0 && s_layerModules.TryGetValue(rest[..dot], out var module))
                {
                    string canonical = $"layers.{layer.Groups[1].Value}.{module}.{rest[(dot + 1)..]}";
                    collector.Add(info.Shape.Rank == 1 ? HfTransforms.Plain(info, canonical) : info.As(canonical));
                    continue;
                }
            }

            collector.AddUnmapped(rawName);
        }

        collector.ThrowIfUnmapped(c.ModelType);
        return collector.Weights;
    }

    // [tables, 1 + gh·gw + detection, hidden] → the same with the patch rows resampled bicubically to grid × grid (class row first, detection rows last, as the token order).
    private static WeightInfo PositionTable(WeightInfo info, string name, int tables, int hidden, int detection, int sourceHeight, int sourceWidth, int grid)
    {
        long sourceRows = 1 + (long)sourceHeight * sourceWidth + detection;
        long rows = 1 + (long)grid * grid + detection;
        return WeightInfo.Derived(name, DataType.F32, tables == 1 && name == "position_embedding.weight" ? new Shape(rows, hidden) : new Shape(tables, rows, hidden), () =>
        {
            var values = HfTransforms.DecodeF32(info);
            var result = new float[tables * rows * hidden];
            float[] my = PositionGridResampling.Bicubic(sourceHeight, grid);
            float[] mx = PositionGridResampling.Bicubic(sourceWidth, grid);
            for (int t = 0; t < tables; t++)
            {
                var source = values.AsSpan((int)(t * sourceRows * hidden), (int)(sourceRows * hidden));
                var target = result.AsSpan((int)(t * rows * hidden), (int)(rows * hidden));
                source[..hidden].CopyTo(target);
                source[(int)((sourceRows - detection) * hidden)..].CopyTo(target[(int)((rows - detection) * hidden)..]);
                var patches = source.Slice(hidden, sourceHeight * sourceWidth * hidden);
                var byRow = new float[grid * sourceWidth * hidden];
                for (int oy = 0; oy < grid; oy++)
                {
                    for (int iy = 0; iy < sourceHeight; iy++)
                    {
                        float w = my[oy * sourceHeight + iy];
                        if (w == 0f)
                        {
                            continue;
                        }

                        for (int i = 0; i < sourceWidth * hidden; i++)
                        {
                            byRow[oy * sourceWidth * hidden + i] += w * patches[iy * sourceWidth * hidden + i];
                        }
                    }
                }

                for (int oy = 0; oy < grid; oy++)
                {
                    for (int ox = 0; ox < grid; ox++)
                    {
                        var destination = target.Slice(hidden + (oy * grid + ox) * hidden, hidden);
                        for (int ix = 0; ix < sourceWidth; ix++)
                        {
                            float w = mx[ox * sourceWidth + ix];
                            if (w == 0f)
                            {
                                continue;
                            }

                            for (int k = 0; k < hidden; k++)
                            {
                                destination[k] += w * byRow[(oy * sourceWidth + ix) * hidden + k];
                            }
                        }
                    }
                }
            }

            return HfTransforms.EncodeF32(result);
        });
    }

    private static string[] Labels(HfConfig c)
    {
        if (c.Get("id2label") is not { ValueKind: JsonValueKind.Object } map)
        {
            return [];
        }

        return [.. map.EnumerateObject().Select(p => (Index: int.Parse(p.Name, System.Globalization.CultureInfo.InvariantCulture), Label: p.Value.GetString() ?? "")).OrderBy(p => p.Index).Select(p => p.Label)];
    }

    [GeneratedRegex(@"^(class_labels_classifier|bbox_predictor)\.layers\.(\d+)\.(weight|bias)$")]
    private static partial Regex HeadLayer();

    [GeneratedRegex(@"^encoder\.layer\.(\d+)\.(.+)$")]
    private static partial Regex Layer();
}
