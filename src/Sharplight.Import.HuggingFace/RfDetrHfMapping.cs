using System.Globalization;
using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Hugging Face RF-DETR (<c>rf_detr</c>: RfDetrForObjectDetection, <c>Roboflow/rf-detr-*</c>) → canonical <c>rf_detr</c> (documented on <c>RfDetrAdapter</c>). The repositories
/// keep the authors' tensor names (<c>backbone.0.encoder.encoder.*</c>, <c>transformer.decoder.*</c>, a fused <c>in_proj</c> in the decoder's self-attention), which transformers renames
/// on load; they are read as they are. Only the first of the 13 query groups (the one inference uses) is mapped: <c>query_feat</c> / <c>refpoint_embed</c> rows [0, queries) and the
/// first <c>enc_*</c> heads. The graph runs on a fixed square input (the processor's size, 704 px for the released models). The segmentation models (<c>RfDetrForInstanceSegmentation</c>)
/// and several feature levels are not supported.
/// </summary>
internal sealed partial class RfDetrHfMapping : IHfArchitectureMapping
{
    /// <inheritdoc/>
    public string Architecture => "rf_detr";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["rf_detr"];

    /// <inheritdoc/>
    public bool NeedsTokenizer => false;

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig c)
    {
        if (c.Architectures.Any(a => a.Contains("Segmentation", StringComparison.Ordinal)))
        {
            yield return "rf_detr: the instance-segmentation models are not supported.";
        }

        if ((c.GetInt("num_feature_levels") ?? 1) != 1)
        {
            yield return "rf_detr: only one feature level is supported.";
        }

        if (c.Get("backbone_config") is not { ValueKind: JsonValueKind.Object } backbone || backbone.TryGetProperty("use_swiglu_ffn", out var swiglu) && swiglu.ValueKind == JsonValueKind.True)
        {
            yield return "rf_detr: only the plain-MLP DINOv2 backbone is supported.";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        string a = Architecture + ".";
        var backbone = c.Get("backbone_config")!.Value;
        int Int(string key, int fallback) => backbone.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : fallback;
        int imageSize = Int("image_size", 704);
        var processor = HfImageProcessorConfig.Read(c.Directory, imageSize);
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "image_size"] = HfImageProcessorConfig.InputSize(processor, imageSize),
            [a + "patch_size"] = Int("patch_size", 16),
            [a + "hidden_size"] = c.Int("d_model"),
            [a + "query_count"] = c.Int("num_queries"),
            [a + "backbone.hidden_size"] = Int("hidden_size", 384),
            [a + "backbone.layer_count"] = Int("num_hidden_layers", 12),
            [a + "backbone.head_count"] = Int("num_attention_heads", 6),
            [a + "backbone.window_count"] = Int("num_windows", 1),
            [a + "backbone.window_blocks"] = backbone.TryGetProperty("window_block_indexes", out var wb) ? string.Join(',', wb.EnumerateArray().Select(v => v.GetInt32())) : "",
            [a + "backbone.out_layers"] = string.Join(',', backbone.GetProperty("out_indices").EnumerateArray().Select(v => v.GetInt32())),
            [a + "backbone.norm_eps"] = backbone.TryGetProperty("layer_norm_eps", out var eps) ? (float)eps.GetDouble() : 1e-6f,
            [a + "projector.block_count"] = c.GetInt("c2f_num_blocks") ?? 3,
            [a + "projector.expansion"] = (float)(c.GetDouble("hidden_expansion") ?? 0.5),
            [a + "decoder.layer_count"] = c.Int("decoder_layers"),
            [a + "decoder.self_head_count"] = c.Int("decoder_self_attention_heads"),
            [a + "decoder.cross_head_count"] = c.Int("decoder_cross_attention_heads"),
            [a + "decoder.point_count"] = c.Int("decoder_n_points"),
            [a + "decoder.ffn_hidden_size"] = c.Int("decoder_ffn_dim"),
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
        int queries = c.Int("num_queries");
        int d = c.Int("d_model");
        var collector = new HfWeightCollector();
        foreach (var (rawName, tensor) in tensors)
        {
            var info = tensor.ToWeightInfo();
            if (rawName.EndsWith(".mask_token", StringComparison.Ordinal))
            {
                continue;
            }

            if (Regex.Match(rawName, @"^(?:enc_out_bbox_embed|transformer\.enc_out_bbox_embed|transformer\.enc_out_class_embed|transformer\.enc_output|transformer\.enc_output_norm)\.(\d+)\.") is { Success: true } group
                && group.Groups[1].Value != "0")
            {
                continue;
            }

            switch (rawName)
            {
                case "query_feat.weight":
                    collector.Add(info.RowSlice("query_feat", 0, queries));
                    continue;
                case "refpoint_embed.weight":
                    collector.Add(info.RowSlice("reference_embed", 0, queries));
                    continue;
                case "backbone.0.encoder.encoder.embeddings.cls_token":
                    collector.Add(info.As("backbone.cls_token", new Tensors.Shape(1, info.Shape.Elements())));
                    continue;
                case "backbone.0.encoder.encoder.embeddings.position_embeddings":
                    long width = info.Shape[info.Shape.Rank - 1].Fixed;
                    collector.Add(info.As("backbone.position_embedding.weight", new Tensors.Shape(info.Shape.Elements() / width, width)));
                    continue;
            }

            if (Regex.Match(rawName, @"^transformer\.decoder\.layers\.(\d+)\.self_attn\.in_proj_(weight|bias)$") is { Success: true } fused)
            {
                string p = $"decoder.layers.{fused.Groups[1].Value}.self_attn.";
                string kind = fused.Groups[2].Value;
                collector.Add(HfTransforms.Plain(info.RowSlice("", 0, d), p + "q." + kind));
                collector.Add(HfTransforms.Plain(info.RowSlice("", d, d), p + "k." + kind));
                collector.Add(HfTransforms.Plain(info.RowSlice("", 2L * d, d), p + "v." + kind));
                continue;
            }

            string? name = Rename(rawName);
            if (name is null)
            {
                collector.AddUnmapped(rawName);
                continue;
            }

            collector.Add(info.Shape.Rank == 1 ? HfTransforms.Plain(info, name) : info.As(name));
        }

        collector.ThrowIfUnmapped(c.ModelType);
        return collector.Weights;
    }

    private static readonly (Regex Pattern, string Replacement)[] s_renames =
    [
        (BackboneLayer(), "backbone.layers.$1.$2"),
        (new Regex(@"^backbone\.0\.encoder\.encoder\.embeddings\.patch_embeddings\.projection\.(weight|bias)$"), "backbone.patch_embed.$1"),
        (new Regex(@"^backbone\.0\.encoder\.encoder\.layernorm\.(weight|bias)$"), "backbone.norm.$1"),
        (new Regex(@"^backbone\.0\.projector\.stages\.0\.0\.cv([12])\.conv\.weight$"), "projector.conv$1.weight"),
        (new Regex(@"^backbone\.0\.projector\.stages\.0\.0\.cv([12])\.bn\.(weight|bias)$"), "projector.conv$1.norm.$2"),
        (new Regex(@"^backbone\.0\.projector\.stages\.0\.0\.m\.(\d+)\.cv([12])\.conv\.weight$"), "projector.bottlenecks.$1.conv$2.weight"),
        (new Regex(@"^backbone\.0\.projector\.stages\.0\.0\.m\.(\d+)\.cv([12])\.bn\.(weight|bias)$"), "projector.bottlenecks.$1.conv$2.norm.$3"),
        (new Regex(@"^backbone\.0\.projector\.stages\.0\.1\.(weight|bias)$"), "projector.norm.$1"),
        (new Regex(@"^bbox_embed\.layers\.(\d+)\.(weight|bias)$"), "decoder.box_head.$1.$2"),
        (new Regex(@"^class_embed\.(weight|bias)$"), "decoder.class_head.$1"),
        (new Regex(@"^transformer\.decoder\.layers\.(\d+)\.self_attn\.out_proj\.(weight|bias)$"), "decoder.layers.$1.self_attn.o.$2"),
        (new Regex(@"^transformer\.decoder\.layers\.(\d+)\.cross_attn\.(.+)$"), "decoder.layers.$1.cross_attn.$2"),
        (new Regex(@"^transformer\.decoder\.layers\.(\d+)\.norm1\.(weight|bias)$"), "decoder.layers.$1.self_attn_norm.$2"),
        (new Regex(@"^transformer\.decoder\.layers\.(\d+)\.norm2\.(weight|bias)$"), "decoder.layers.$1.cross_attn_norm.$2"),
        (new Regex(@"^transformer\.decoder\.layers\.(\d+)\.norm3\.(weight|bias)$"), "decoder.layers.$1.ffn_norm.$2"),
        (new Regex(@"^transformer\.decoder\.layers\.(\d+)\.linear1\.(weight|bias)$"), "decoder.layers.$1.ffn.up.$2"),
        (new Regex(@"^transformer\.decoder\.layers\.(\d+)\.linear2\.(weight|bias)$"), "decoder.layers.$1.ffn.down.$2"),
        (new Regex(@"^transformer\.decoder\.norm\.(weight|bias)$"), "decoder.norm.$1"),
        (new Regex(@"^transformer\.decoder\.ref_point_head\.layers\.(\d+)\.(weight|bias)$"), "decoder.query_pos.$1.$2"),
        (new Regex(@"^transformer\.enc_out_bbox_embed\.0\.layers\.(\d+)\.(weight|bias)$"), "enc_box_head.$1.$2"),
        (new Regex(@"^transformer\.enc_out_class_embed\.0\.(weight|bias)$"), "enc_class_head.$1"),
        (new Regex(@"^transformer\.enc_output\.0\.(weight|bias)$"), "enc_output.$1"),
        (new Regex(@"^transformer\.enc_output_norm\.0\.(weight|bias)$"), "enc_output_norm.$1"),
    ];

    private static string? Rename(string name)
    {
        foreach (var (pattern, replacement) in s_renames)
        {
            if (pattern.Match(name) is { Success: true } m)
            {
                string result = m.Result(replacement);
                return result
                    .Replace(".attention.attention.query.", ".attn.q.", StringComparison.Ordinal)
                    .Replace(".attention.attention.key.", ".attn.k.", StringComparison.Ordinal)
                    .Replace(".attention.attention.value.", ".attn.v.", StringComparison.Ordinal)
                    .Replace(".attention.output.dense.", ".attn.o.", StringComparison.Ordinal)
                    .Replace(".layer_scale1.lambda1", ".ls1.weight", StringComparison.Ordinal)
                    .Replace(".layer_scale2.lambda1", ".ls2.weight", StringComparison.Ordinal)
                    .Replace(".mlp.fc1.", ".mlp.up.", StringComparison.Ordinal)
                    .Replace(".mlp.fc2.", ".mlp.down.", StringComparison.Ordinal);
            }
        }

        return null;
    }

    private static string[] Labels(HfConfig c)
    {
        if (c.Get("id2label") is not { ValueKind: JsonValueKind.Object } map)
        {
            return [];
        }

        return [.. map.EnumerateObject().Select(p => (Index: int.Parse(p.Name, CultureInfo.InvariantCulture), Label: p.Value.GetString() ?? "")).OrderBy(p => p.Index).Select(p => p.Label)];
    }

    [GeneratedRegex(@"^backbone\.0\.encoder\.encoder\.encoder\.layer\.(\d+)\.(.+)$")]
    private static partial Regex BackboneLayer();
}
