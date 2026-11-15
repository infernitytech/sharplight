using System.Globalization;
using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Hugging Face RT-DETR (<c>rt_detr</c>: RTDetrForObjectDetection with the ResNet-D backbone) → canonical <c>rt_detr</c> (documented on
/// <c>RtDetrAdapter</c>). Every convolution is stored with its batch norm folded in as <c>name.weight</c> and <c>name.bias</c>:
/// <c>backbone.stem.i</c>, <c>backbone.stages.s.b.conv{k}</c> / <c>.short</c>, <c>encoder_proj.i</c>, <c>decoder_proj.i</c>, <c>hybrid.lateral.i</c>,
/// <c>hybrid.down.i</c>, <c>hybrid.{fpn,pan}.i.conv{1,2,3}</c> and <c>hybrid.{fpn,pan}.i.rep.j.conv{1,2}</c>. The remaining weights:
/// <c>aifi.l.</c> and <c>decoder.layers.i.</c> (attention <c>self_attn.{q,k,v,o}</c>, deformable <c>cross_attn.{sampling_offsets,attention_weights,
/// value_proj,output_proj}</c>, <c>*_norm</c>, <c>ffn.{up,down}</c>), <c>decoder.{box_head.i.j,class_head.i,query_pos.j}</c>, <c>enc_output.{linear,norm}</c>,
/// <c>enc_score_head</c>, <c>enc_box_head.j</c>. The graph runs on a fixed square input (the processor's size, 640 px).
/// </summary>
internal sealed partial class RtDetrHfMapping : IHfArchitectureMapping
{
    /// <inheritdoc/>
    public string Architecture => "rt_detr";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["rt_detr"];

    /// <inheritdoc/>
    public bool NeedsTokenizer => false;

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig c)
    {
        if (c.Get("backbone_config") is not { ValueKind: JsonValueKind.Object } backbone || backbone.GetProperty("model_type").GetString() != "rt_detr_resnet")
        {
            yield return "rt_detr: only the RT-DETR ResNet backbone is supported.";
        }

        if (c.GetBool("with_box_refine") == false)
        {
            yield return "rt_detr: without box refinement is not supported.";
        }

        if (c.Get("encode_proj_layers") is { ValueKind: JsonValueKind.Array } layers && layers.GetArrayLength() != 1)
        {
            yield return "rt_detr: exactly one AIFI level is supported.";
        }

        if (c.GetBool("normalize_before") == true)
        {
            yield return "rt_detr: pre-norm encoder layers are not supported.";
        }

        if (c.GetBool("learn_initial_query") == true)
        {
            yield return "rt_detr: learned initial queries are not supported.";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        string a = Architecture + ".";
        var backbone = c.Get("backbone_config")!.Value;
        var processor = HfImageProcessorConfig.Read(c.Directory, 640);
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "hidden_size"] = c.Int("d_model"),
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
            [a + "image_size"] = HfImageProcessorConfig.InputSize(processor, 640),
            [a + "level_count"] = c.GetInt("num_feature_levels") ?? 3,
            [a + "strides"] = string.Join(',', c.Get("feat_strides")!.Value.EnumerateArray().Select(v => v.GetInt32())),
            [a + "backbone.depths"] = string.Join(',', backbone.GetProperty("depths").EnumerateArray().Select(v => v.GetInt32())),
            [a + "norm_eps"] = (float)(c.GetDouble("layer_norm_eps") ?? 1e-5),
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
                SafetensorsTensor Norm(string suffix) => tensors[conv.Norm + suffix];
                var (w, b) = HfBatchNormFolding.Fold(info, Norm(".weight").ToWeightInfo(), Norm(".bias").ToWeightInfo(),
                    Norm(".running_mean").ToWeightInfo(), Norm(".running_var").ToWeightInfo(), eps, conv.Canonical);
                collector.Add(w);
                collector.Add(b);
                continue;
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

    private static bool IsBatchNorm(string name)
        => name.Contains(".normalization.", StringComparison.Ordinal) || BatchNormInSequence().IsMatch(name) || EncoderNorm().IsMatch(name);

    // A convolution weight, the name of its batch norm and the canonical name of the folded convolution.
    private static (string Norm, string Canonical)? ConvName(string name)
    {
        if (Match(@"^model\.backbone\.model\.embedder\.embedder\.(\d+)\.convolution\.weight$", name) is { } stem)
        {
            return (name.Replace(".convolution.weight", ".normalization", StringComparison.Ordinal), $"backbone.stem.{stem[0]}");
        }

        if (Match(@"^model\.backbone\.model\.encoder\.stages\.(\d+)\.layers\.(\d+)\.layer\.(\d+)\.convolution\.weight$", name) is { } layer)
        {
            return (name.Replace(".convolution.weight", ".normalization", StringComparison.Ordinal),
                $"backbone.stages.{layer[0]}.{layer[1]}.conv{int.Parse(layer[2], CultureInfo.InvariantCulture) + 1}");
        }

        if (Match(@"^model\.backbone\.model\.encoder\.stages\.(\d+)\.layers\.(\d+)\.shortcut(?:\.1)?\.convolution\.weight$", name) is { } shortcut)
        {
            return (name.Replace(".convolution.weight", ".normalization", StringComparison.Ordinal), $"backbone.stages.{shortcut[0]}.{shortcut[1]}.short");
        }

        if (Match(@"^model\.(encoder|decoder)_input_proj\.(\d+)\.0\.weight$", name) is { } proj)
        {
            return (name.Replace(".0.weight", ".1", StringComparison.Ordinal), $"{proj[0]}_proj.{proj[1]}");
        }

        if (Match(@"^model\.encoder\.(.+)\.conv\.weight$", name) is { } hybrid)
        {
            string path = hybrid[0];
            string canonical = path.StartsWith("lateral_convs.", StringComparison.Ordinal) ? "hybrid.lateral." + path["lateral_convs.".Length..]
                : path.StartsWith("downsample_convs.", StringComparison.Ordinal) ? "hybrid.down." + path["downsample_convs.".Length..]
                : "hybrid." + path.Replace("fpn_blocks.", "fpn.", StringComparison.Ordinal).Replace("pan_blocks.", "pan.", StringComparison.Ordinal)
                    .Replace("bottlenecks.", "rep.", StringComparison.Ordinal);
            return ($"model.encoder.{path}.norm", canonical);
        }

        return null;
    }

    private static string[]? Match(string pattern, string input)
    {
        var m = Regex.Match(input, pattern);
        return m.Success ? [.. m.Groups.Values.Skip(1).Select(g => g.Value)] : null;
    }

    private static string? Rename(string name)
    {
        if (LayerRaw().Match(name) is { Success: true } layer)
        {
            string group = layer.Groups[1].Success ? $"aifi.{layer.Groups[2].Value}" : $"decoder.layers.{layer.Groups[2].Value}";
            return s_layerModules.TryGetValue(layer.Groups[3].Value, out var module) ? $"{group}.{module}.{layer.Groups[4].Value}" : null;
        }

        if (Regex.Match(name, @"^model\.decoder\.bbox_embed\.(\d+)\.layers\.(\d+)\.(weight|bias)$") is { Success: true } box)
        {
            return $"decoder.box_head.{box.Groups[1].Value}.{box.Groups[2].Value}.{box.Groups[3].Value}";
        }

        if (Regex.Match(name, @"^model\.decoder\.class_embed\.(\d+)\.(weight|bias)$") is { Success: true } cls)
        {
            return $"decoder.class_head.{cls.Groups[1].Value}.{cls.Groups[2].Value}";
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
