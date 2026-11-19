using System.Globalization;
using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Hugging Face PP-DocLayoutV3 (<c>pp_doclayout_v3</c>: PPDocLayoutV3ForObjectDetection with an HGNetV2 backbone) → canonical <c>pp_doclayout_v3</c> (documented on
/// <c>PpDocLayoutAdapter</c>). Every convolution with a batch norm is stored folded as <c>name.weight</c> and <c>name.bias</c>; the encoder, decoder and head
/// tensors keep the RT-DETR names (<see cref="RtDetrHfMapping"/>). The decoder's class and box refinement heads are tied to <c>enc_score_head</c> and <c>enc_bbox_head</c>
/// (the checkpoint stores only those). The graph runs on a fixed square input (the processor's size, 800 px).
/// </summary>
internal sealed partial class PpDocLayoutHfMapping : IHfArchitectureMapping
{
    /// <inheritdoc/>
    public string Architecture => "pp_doclayout_v3";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["pp_doclayout_v3"];

    /// <inheritdoc/>
    public bool NeedsTokenizer => false;

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig c)
    {
        if (c.Get("backbone_config") is not { ValueKind: JsonValueKind.Object } backbone || backbone.GetProperty("model_type").GetString() != "hgnet_v2")
        {
            yield return "pp_doclayout_v3: only the HGNetV2 backbone is supported.";
        }
        else if ((backbone.TryGetProperty("arch", out var arch) ? arch.GetString() : "L") != "L" && !backbone.TryGetProperty("stage_in_channels", out _))
        {
            yield return "pp_doclayout_v3: only the L backbone is supported (its stage tables are the defaults).";
        }
        else if (backbone.TryGetProperty("use_learnable_affine_block", out var lab) && lab.ValueKind == JsonValueKind.True)
        {
            yield return "pp_doclayout_v3: learnable affine blocks in the backbone are not supported.";
        }

        if (c.Get("encode_proj_layers") is { ValueKind: JsonValueKind.Array } layers && layers.GetArrayLength() != 1)
        {
            yield return "pp_doclayout_v3: exactly one AIFI level is supported.";
        }

        if (c.GetBool("normalize_before") == true)
        {
            yield return "pp_doclayout_v3: pre-norm encoder layers are not supported.";
        }

        if (c.GetBool("learn_initial_query") == true)
        {
            yield return "pp_doclayout_v3: learned initial queries are not supported.";
        }

        if (c.GetBool("mask_enhanced") == false)
        {
            yield return "pp_doclayout_v3: without mask-enhanced reference boxes is not supported.";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        string a = Architecture + ".";
        var processor = HfImageProcessorConfig.Read(c.Directory, 800);
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "hidden_size"] = c.Int("d_model"),
            [a + "encoder.hidden_size"] = c.GetInt("encoder_hidden_dim") ?? c.Int("d_model"),
            [a + "attention.head_count"] = c.Int("decoder_attention_heads"),
            [a + "aifi.layer_count"] = c.Int("encoder_layers"),
            [a + "aifi.ffn_hidden_size"] = c.Int("encoder_ffn_dim"),
            [a + "aifi.mlp.activation"] = c.GetString("encoder_activation_function") ?? "gelu",
            [a + "decoder.layer_count"] = c.Int("decoder_layers"),
            [a + "decoder.ffn_hidden_size"] = c.Int("decoder_ffn_dim"),
            [a + "decoder.mlp.activation"] = c.GetString("decoder_activation_function") ?? "relu",
            [a + "decoder.point_count"] = c.Int("decoder_n_points"),
            [a + "mlp.activation"] = c.GetString("activation_function") ?? "silu",
            [a + "query_count"] = c.Int("num_queries"),
            [a + "image_size"] = HfImageProcessorConfig.InputSize(processor, 800),
            [a + "level_count"] = c.GetInt("num_feature_levels") ?? 3,
            [a + "strides"] = string.Join(',', (c.Get("feat_strides") ?? c.Get("feature_strides"))!.Value.EnumerateArray().Select(v => v.GetInt32())),
            [a + "norm_eps"] = (float)(c.GetDouble("layer_norm_eps") ?? 1e-5),
            [a + "prototype_count"] = c.GetInt("num_prototypes") ?? 32,
            [a + "pointer.head_size"] = c.GetInt("global_pointer_head_size") ?? 64,
            [CanonicalKeys.ImageProcessor] = processor.ToJson().ToJsonString(),
        };
        if (c.Get("backbone_config") is { } backbone)
        {
            AddStageTable(md, a + "backbone.stage_in_channels", backbone, "stage_in_channels");
            AddStageTable(md, a + "backbone.stage_mid_channels", backbone, "stage_mid_channels");
            AddStageTable(md, a + "backbone.stage_out_channels", backbone, "stage_out_channels");
            AddStageTable(md, a + "backbone.stage_blocks", backbone, "stage_num_blocks");
            AddStageTable(md, a + "backbone.stage_layers", backbone, "stage_numb_of_layers");
            AddStageTable(md, a + "backbone.stage_downsample", backbone, "stage_downsample");
            AddStageTable(md, a + "backbone.stage_light", backbone, "stage_light_block");
            AddStageTable(md, a + "backbone.stage_kernels", backbone, "stage_kernel_size");
            AddStageTable(md, a + "backbone.stem_channels", backbone, "stem_channels");
        }

        if (Labels(c) is { Length: > 0 } labels)
        {
            md[CanonicalKeys.ClassifierLabels] = labels;
        }

        return md;
    }

    // A table the backbone configuration lists (numbers or booleans), as a comma list; absent tables keep the adapter's L defaults.
    private static void AddStageTable(Dictionary<string, object> md, string key, JsonElement backbone, string property)
    {
        if (backbone.TryGetProperty(property, out var values) && values.ValueKind == JsonValueKind.Array)
        {
            md[key] = string.Join(',', values.EnumerateArray().Select(v => v.ValueKind switch
            {
                JsonValueKind.True => "1",
                JsonValueKind.False => "0",
                _ => v.GetInt32().ToString(CultureInfo.InvariantCulture),
            }));
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        float eps = (float)(c.GetDouble("batch_norm_eps") ?? 1e-5);
        var collector = new HfWeightCollector();
        foreach (var (rawName, tensor) in tensors)
        {
            if (rawName.EndsWith("num_batches_tracked", StringComparison.Ordinal) || rawName.Contains("denoising_class_embed", StringComparison.Ordinal) || IsBatchNorm(rawName))
            {
                continue;
            }

            var info = tensor.ToWeightInfo();
            if (info.Shape.Rank == 4 && ConvName(rawName) is { } conv)
            {
                if (!tensors.ContainsKey(conv.Norm + ".weight"))
                {
                    // A plain convolution with a bias (the prototype projection).
                    collector.Add(info.As(conv.Canonical + ".weight"));
                    collector.Add(HfTransforms.Plain(tensors[rawName.Replace(".weight", ".bias", StringComparison.Ordinal)].ToWeightInfo(), conv.Canonical + ".bias"));
                    continue;
                }

                SafetensorsTensor Norm(string suffix) => tensors[conv.Norm + suffix];
                var (w, b) = HfBatchNormFolding.Fold(info, Norm(".weight").ToWeightInfo(), Norm(".bias").ToWeightInfo(),
                    Norm(".running_mean").ToWeightInfo(), Norm(".running_var").ToWeightInfo(), eps, conv.Canonical);
                collector.Add(w);
                collector.Add(b);
                continue;
            }

            if (rawName == "model.encoder.encoder_mask_output.conv.bias")
            {
                continue;   // read with its weight
            }

            if (Rename(rawName) is not { } canonical)
            {
                collector.AddUnmapped(rawName);
                continue;
            }

            collector.Add(info.Shape.Rank == 1 ? HfTransforms.Plain(info, canonical) : info.As(canonical));
        }

        collector.ThrowIfUnmapped(c.ModelType);
        return collector.Weights;
    }

    // Batch norm tensors (consumed by the folding): every ".normalization." / ".norm." sibling of a convolution and the norms of the projection sequences.
    private static bool IsBatchNorm(string name)
        => name.Contains(".normalization.", StringComparison.Ordinal) || BatchNormInSequence().IsMatch(name) || EncoderNorm().IsMatch(name);

    // A convolution weight, the name of its batch norm and the canonical name of the folded convolution.
    private static (string Norm, string Canonical)? ConvName(string name)
    {
        string norm = name.Replace(".convolution.weight", ".normalization", StringComparison.Ordinal);
        if (Regex.Match(name, @"^model\.backbone\.model\.embedder\.(stem\w+)\.convolution\.weight$") is { Success: true } stem)
        {
            return (norm, "backbone.stem." + stem.Groups[1].Value);
        }

        if (Regex.Match(name, @"^model\.backbone\.model\.encoder\.stages\.(\d+)\.downsample\.convolution\.weight$") is { Success: true } down)
        {
            return (norm, $"backbone.stages.{down.Groups[1].Value}.down");
        }

        if (Regex.Match(name, @"^model\.backbone\.model\.encoder\.stages\.(\d+)\.blocks\.(\d+)\.layers\.(\d+)(\.conv[12])?\.convolution\.weight$") is { Success: true } layer)
        {
            return (norm, $"backbone.stages.{layer.Groups[1].Value}.{layer.Groups[2].Value}.layers.{layer.Groups[3].Value}{layer.Groups[4].Value}");
        }

        if (Regex.Match(name, @"^model\.backbone\.model\.encoder\.stages\.(\d+)\.blocks\.(\d+)\.aggregation\.(\d+)\.convolution\.weight$") is { Success: true } agg)
        {
            return (norm, $"backbone.stages.{agg.Groups[1].Value}.{agg.Groups[2].Value}.agg.{agg.Groups[3].Value}");
        }

        if (Regex.Match(name, @"^model\.(encoder|decoder)_input_proj\.(\d+)\.0\.weight$") is { Success: true } proj)
        {
            return (name.Replace(".0.weight", ".1", StringComparison.Ordinal), $"{proj.Groups[1].Value}_proj.{proj.Groups[2].Value}");
        }

        if (Regex.Match(name, @"^model\.encoder\.mask_feature_head\.scale_heads\.(\d+)\.layers\.(\d+)\.convolution\.weight$") is { Success: true } scale)
        {
            return (norm, $"mask_head.scale.{scale.Groups[1].Value}.{scale.Groups[2].Value}");
        }

        return name switch
        {
            "model.encoder.mask_feature_head.output_conv.convolution.weight" => (norm, "mask_head.out"),
            "model.encoder.encoder_mask_lateral.convolution.weight" => (norm, "mask_lateral"),
            "model.encoder.encoder_mask_output.base_conv.convolution.weight" => (norm, "mask_out.base"),
            "model.encoder.encoder_mask_output.conv.weight" => ("", "mask_out.proj"),
            _ => HybridConv(name),
        };
    }

    private static (string Norm, string Canonical)? HybridConv(string name)
    {
        if (Regex.Match(name, @"^model\.encoder\.(.+)\.conv\.weight$") is not { Success: true } hybrid)
        {
            return null;
        }

        string path = hybrid.Groups[1].Value;
        string canonical = path.StartsWith("lateral_convs.", StringComparison.Ordinal) ? "hybrid.lateral." + path["lateral_convs.".Length..]
            : path.StartsWith("downsample_convs.", StringComparison.Ordinal) ? "hybrid.down." + path["downsample_convs.".Length..]
            : "hybrid." + path.Replace("fpn_blocks.", "fpn.", StringComparison.Ordinal).Replace("pan_blocks.", "pan.", StringComparison.Ordinal)
                .Replace("bottlenecks.", "rep.", StringComparison.Ordinal);
        return ($"model.encoder.{path}.norm", canonical);
    }

    private static string? Rename(string name)
    {
        if (LayerRaw().Match(name) is { Success: true } layer)
        {
            string group = layer.Groups[1].Success ? $"aifi.{layer.Groups[2].Value}" : $"decoder.layers.{layer.Groups[2].Value}";
            return s_layerModules.TryGetValue(layer.Groups[3].Value, out var module) ? $"{group}.{module}.{layer.Groups[4].Value}" : null;
        }

        if (Regex.Match(name, @"^model\.decoder\.query_pos_head\.layers\.(\d+)\.(weight|bias)$") is { Success: true } pos)
        {
            return $"decoder.query_pos.{pos.Groups[1].Value}.{pos.Groups[2].Value}";
        }

        if (Regex.Match(name, @"^model\.enc_output\.(0|1)\.(weight|bias)$") is { Success: true } output)
        {
            return $"enc_output.{(output.Groups[1].Value == "0" ? "linear" : "norm")}.{output.Groups[2].Value}";
        }

        if (Regex.Match(name, @"^model\.enc_bbox_head\.layers\.(\d+)\.(weight|bias)$") is { Success: true } encBox)
        {
            return $"enc_box_head.{encBox.Groups[1].Value}.{encBox.Groups[2].Value}";
        }

        if (Regex.Match(name, @"^model\.mask_query_head\.layers\.(\d+)\.(weight|bias)$") is { Success: true } mask)
        {
            return $"mask_query.{mask.Groups[1].Value}.{mask.Groups[2].Value}";
        }

        if (Regex.Match(name, @"^model\.decoder_order_head\.(\d+)\.(weight|bias)$") is { Success: true } order)
        {
            return $"decoder.order_head.{order.Groups[1].Value}.{order.Groups[2].Value}";
        }

        if (Regex.Match(name, @"^model\.decoder_global_pointer\.dense\.(weight|bias)$") is { Success: true } pointer)
        {
            return $"decoder.global_pointer.{pointer.Groups[1].Value}";
        }

        if (Regex.Match(name, @"^model\.decoder_norm\.(weight|bias)$") is { Success: true } norm)
        {
            return $"decoder_norm.{norm.Groups[1].Value}";
        }

        return name.StartsWith("model.enc_score_head.", StringComparison.Ordinal) ? name["model.".Length..] : null;
    }

    private static string[] Labels(HfConfig c)
    {
        if (c.Get("id2label") is not { ValueKind: JsonValueKind.Object } map)
        {
            return [];
        }

        return [.. map.EnumerateObject().Select(p => (Index: int.Parse(p.Name, CultureInfo.InvariantCulture), Label: p.Value.GetString() ?? "")).OrderBy(p => p.Index).Select(p => p.Label)];
    }

    [GeneratedRegex(@"^model\.(?:encoder\.encoder\.(\d+)|decoder)\.layers\.(\d+)\.(.+)\.(weight|bias)$")]
    private static partial Regex LayerRaw();

    [GeneratedRegex(@"^model\.(encoder|decoder)_input_proj\.\d+\.1\.")]
    private static partial Regex BatchNormInSequence();

    [GeneratedRegex(@"^model\.encoder\.(?!encoder\.).+\.norm\.")]
    private static partial Regex EncoderNorm();

    private static readonly Dictionary<string, string> s_layerModules = new(StringComparer.Ordinal)
    {
        ["self_attn.q_proj"] = "self_attn.q",
        ["self_attn.k_proj"] = "self_attn.k",
        ["self_attn.v_proj"] = "self_attn.v",
        ["self_attn.out_proj"] = "self_attn.o",
        ["self_attn.o_proj"] = "self_attn.o",
        ["encoder_attn.sampling_offsets"] = "cross_attn.sampling_offsets",
        ["encoder_attn.attention_weights"] = "cross_attn.attention_weights",
        ["encoder_attn.value_proj"] = "cross_attn.value_proj",
        ["encoder_attn.output_proj"] = "cross_attn.output_proj",
        ["self_attn_layer_norm"] = "self_attn_norm",
        ["encoder_attn_layer_norm"] = "cross_attn_norm",
        ["final_layer_norm"] = "ffn_norm",
        ["fc1"] = "ffn.up",
        ["fc2"] = "ffn.down",
    };
}
