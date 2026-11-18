using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;
using Sharplight.Vision;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Hugging Face plain vision transformers (<c>vit</c>: ViTModel / ViTForImageClassification, <c>dinov2</c> and
/// <c>dinov2_with_registers</c>: Dinov2Model, DINOv2 with LayerScale and register tokens; <c>dinov3_vit</c>: DINOv3ViTModel, the same with
/// two-dimensional rotary positions on the patch rows instead of a position table and optionally gated MLPs) → canonical <c>vit</c> (documented on
/// <c>ImageEncoderAdapter</c>). The patch convolution is read as a matrix product (zero-copy reshape); the position table is resampled
/// bicubically (as PyTorch does at run time) to the grid of the processor's input size, so DINOv2 checkpoints trained at 518 px run at the
/// 224 px of their processor with the reference's numbers; the processor configuration, labels and registers are stored as canonical metadata.
/// </summary>
internal sealed partial class VitHfMapping : IHfArchitectureMapping
{
    /// <inheritdoc/>
    public string Architecture => "vit";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["vit", "dinov2", "dinov2_with_registers", "dinov3_vit"];

    /// <inheritdoc/>
    public bool NeedsTokenizer => false;

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig c)
    {
        if (c.GetBool("use_swiglu_ffn") == true)
        {
            yield return "dinov2: the SwiGLU feed-forward (giant models) is not supported.";
        }

        if (c.GetBool("apply_layernorm") == false)
        {
            yield return "dinov2: apply_layernorm=false is not supported.";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        string a = Architecture + ".";
        int hidden = c.Int("hidden_size");
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = c.Int("num_hidden_layers"),
            [a + "hidden_size"] = hidden,
            [a + "ffn_hidden_size"] = c.GetInt("intermediate_size") ?? (int)Math.Round(hidden * (c.GetDouble("mlp_ratio") ?? 4.0)),
            [a + "attention.head_count"] = c.Int("num_attention_heads"),
            [a + "norm_eps"] = (float)(c.GetDouble("layer_norm_eps") ?? 1e-12),
            [a + "mlp.activation"] = (c.GetString("hidden_act") ?? "gelu") switch
            {
                "gelu" => "gelu",
                "gelu_new" or "gelu_pytorch_tanh" or "gelu_fast" => "gelu_tanh",
                "quick_gelu" => "quick_gelu",
                "relu" => "relu",
                "silu" or "swish" => "silu",
                var other => throw new NotSupportedException($"{c.ModelType}: activation '{other}' is not supported."),
            },
            [a + "patch_size"] = c.Int("patch_size"),
            [a + "image_size"] = InputSize(c, out var processor),
            [a + "register_count"] = c.GetInt("num_register_tokens") ?? 0,
            [a + "pooling"] = "cls",
            [CanonicalKeys.ImageProcessor] = processor.ToJson().ToJsonString(),
        };
        if (c.ModelType == "dinov3_vit")
        {
            md[a + "rope_theta"] = (float)(c.GetDouble("rope_theta") ?? 100.0);
        }

        if (Labels(c) is { Length: > 0 } labels)
        {
            md[CanonicalKeys.ClassifierLabels] = labels;
            md[CanonicalKeys.ClassifierActivation] = "softmax";
        }

        return md;
    }

    private static int InputSize(HfConfig c, out Imaging.ImageProcessorConfig processor)
    {
        int configured = c.GetInt("image_size") ?? 224;
        processor = HfImageProcessorConfig.Read(c.Directory, configured);
        return HfImageProcessorConfig.InputSize(processor, configured);
    }

    private static string[] Labels(HfConfig c)
    {
        if (c.Get("id2label") is not { ValueKind: JsonValueKind.Object } map)
        {
            return [];
        }

        return [.. map.EnumerateObject().Select(p => (Index: int.Parse(p.Name, System.Globalization.CultureInfo.InvariantCulture), Label: p.Value.GetString() ?? "")).OrderBy(p => p.Index).Select(p => p.Label)];
    }

    [GeneratedRegex(@"^(?:encoder\.)?layer\.(\d+)\.(.+)$")]
    private static partial Regex LayerName();

    private static readonly Dictionary<string, string> s_layerModules = new(StringComparer.Ordinal)
    {
        ["layernorm_before"] = "input_norm",
        ["norm1"] = "input_norm",
        ["layernorm_after"] = "post_attention_norm",
        ["norm2"] = "post_attention_norm",
        ["attention.attention.query"] = "attention.q",
        ["attention.attention.key"] = "attention.k",
        ["attention.attention.value"] = "attention.v",
        ["attention.output.dense"] = "attention.o",
        ["attention.q_proj"] = "attention.q",
        ["attention.k_proj"] = "attention.k",
        ["attention.v_proj"] = "attention.v",
        ["attention.o_proj"] = "attention.o",
        ["mlp.up_proj"] = "mlp.up",
        ["mlp.down_proj"] = "mlp.down",
        ["mlp.gate_proj"] = "mlp.gate",
        ["intermediate.dense"] = "mlp.up",
        ["mlp.fc1"] = "mlp.up",
        ["output.dense"] = "mlp.down",
        ["mlp.fc2"] = "mlp.down",
    };

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        int hidden = c.Int("hidden_size");
        int patch = c.Int("patch_size");
        int imageSize = InputSize(c, out _);
        var collector = new HfWeightCollector();
        foreach (var (rawName, tensor) in tensors)
        {
            var info = tensor.ToWeightInfo();
            string name = rawName;
            foreach (string prefix in new[] { "vit.", "dinov2.", "model." })
            {
                if (name.StartsWith(prefix, StringComparison.Ordinal))
                {
                    name = name[prefix.Length..];
                    break;
                }
            }

            switch (name)
            {
                case "embeddings.cls_token":
                    collector.Add(HfTransforms.ToF32(info, "class_embedding", new Shape(hidden)));
                    continue;
                case "embeddings.register_tokens":
                    collector.Add(HfTransforms.ToF32(info, "register_embedding.weight", new Shape(info.Shape.Elements() / hidden, hidden)));
                    continue;
                case "embeddings.mask_token" or "pooler.dense.weight" or "pooler.dense.bias":
                    continue;
                case "embeddings.patch_embeddings.projection.weight" or "embeddings.patch_embeddings.weight":
                    collector.Add(info.As("patch_embed.weight", new Shape(hidden, info.Shape.Elements() / hidden)));
                    continue;
                case "embeddings.patch_embeddings.projection.bias" or "embeddings.patch_embeddings.bias":
                    collector.Add(HfTransforms.Plain(info, "patch_embed.bias"));
                    continue;
                case "embeddings.position_embeddings":
                    collector.Add(PositionEmbedding(info, hidden, imageSize / patch, c.GetBool("interpolate_antialias") == true));
                    continue;
                case "layernorm.weight" or "layernorm.bias" or "norm.weight" or "norm.bias":
                    collector.Add(HfTransforms.Plain(info, "output_norm." + name[(name.IndexOf('.', StringComparison.Ordinal) + 1)..]));
                    continue;
                case "classifier.weight" or "classifier.bias":
                    collector.Add(info.Shape.Rank == 1 ? HfTransforms.Plain(info, "classifier.out.bias") : info.As("classifier.out.weight"));
                    continue;
            }

            var m = LayerName().Match(name);
            if (m.Success && m.Groups[2].Value.EndsWith(".lambda1", StringComparison.Ordinal))
            {
                string scale = m.Groups[2].Value.StartsWith("layer_scale1", StringComparison.Ordinal) ? "layer_scale_1" : "layer_scale_2";
                collector.Add(HfTransforms.ToF32(info, $"layers.{m.Groups[1].Value}.{scale}.weight"));
                continue;
            }

            if (m.Success)
            {
                string rest = m.Groups[2].Value;
                int dot = rest.LastIndexOf('.');
                if (dot > 0 && s_layerModules.TryGetValue(rest[..dot], out var module))
                {
                    string canonical = $"layers.{m.Groups[1].Value}.{module}.{rest[(dot + 1)..]}";
                    collector.Add(info.Shape.Rank == 1 ? HfTransforms.Plain(info, canonical) : info.As(canonical));
                    continue;
                }
            }

            collector.AddUnmapped(rawName);
        }

        collector.ThrowIfUnmapped(c.ModelType);
        return collector.Weights;
    }

    // [1, 1 + g², hidden] → [1 + grid², hidden]: the class row is kept, the patch grid is resampled when the input size differs.
    private static WeightInfo PositionEmbedding(WeightInfo info, int hidden, int grid, bool antialias)
    {
        long rows = info.Shape.Elements() / hidden;
        int sourceGrid = (int)Math.Round(Math.Sqrt(rows - 1));
        if (sourceGrid == grid)
        {
            return HfTransforms.ToF32(info, "position_embedding.weight", new Shape(rows, hidden));
        }

        return WeightInfo.Derived("position_embedding.weight", DataType.F32, new Shape(1 + (long)grid * grid, hidden), () =>
        {
            var values = HfTransforms.DecodeF32(info);
            var patches = PositionGridResampling.ResizeGrid(values.AsSpan(hidden, values.Length - hidden), sourceGrid, grid, hidden, antialias);
            var all = new float[hidden + patches.Length];
            values.AsSpan(0, hidden).CopyTo(all);
            patches.CopyTo(all.AsSpan(hidden));
            return HfTransforms.EncodeF32(all);
        });
    }
}
