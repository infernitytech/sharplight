using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Hugging Face SAM 2 / SAM 2.1 (<c>sam2</c>: Sam2Model, <c>sam2_video</c>: the video checkpoints of the same family, of which only the image path is used) →
/// canonical <c>sam2</c> (documented on <c>Sam2Adapter</c>). The weights keep their names; the video memory modules (memory attention, memory encoder, object
/// pointers, temporal encodings) and the mask-prompt encoder are not imported. The graphs run on the square input of the stored processor (1024 px).
/// </summary>
internal sealed class Sam2HfMapping : IHfArchitectureMapping
{
    private static readonly string[] s_ignored =
    [
        "memory_", "mask_downsample.", "object_pointer_proj.", "temporal_positional_encoding_projection_layer.",
        "no_memory_positional_encoding", "no_object_pointer", "occlusion_spatial_embedding_parameter", "prompt_encoder.mask_embed.", "prompt_encoder.shared_embedding.",
    ];

    private static readonly string[] s_kept =
    [
        "vision_encoder.backbone.", "vision_encoder.neck.convs.", "mask_decoder.", "prompt_encoder.", "shared_image_embedding.positional_embedding", "no_memory_embedding",
    ];

    private static readonly string[] s_dense =
    [
        "pos_embed", "pos_embed_window", "positional_embedding", "no_memory_embedding", "mask_decoder.mask_tokens.weight", "mask_decoder.iou_token.weight",
        "mask_decoder.obj_score_token.weight", "prompt_encoder.no_mask_embed.weight", "prompt_encoder.point_embed.weight", "prompt_encoder.not_a_point_embed.weight",
    ];

    /// <inheritdoc/>
    public string Architecture => "sam2";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["sam2", "sam2_video"];

    /// <inheritdoc/>
    public bool NeedsTokenizer => false;

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig c)
    {
        if (c.Get("vision_config") is not { ValueKind: JsonValueKind.Object } vision || !vision.TryGetProperty("backbone_config", out _))
        {
            yield return "sam2: the checkpoint has no vision_config.backbone_config.";
            yield break;
        }

        if (c.Get("mask_decoder_config") is { } decoder && decoder.TryGetProperty("iou_head_depth", out var depth) && depth.GetInt32() != 3)
        {
            yield return "sam2: only an IoU head of depth 3 is supported.";
        }

        if (vision.TryGetProperty("num_feature_levels", out var levels) && levels.GetInt32() != 3)
        {
            yield return "sam2: only three feature levels are supported.";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        string a = Architecture + ".";
        var vision = c.Get("vision_config")!.Value;
        var backbone = vision.GetProperty("backbone_config");
        var decoder = c.Get("mask_decoder_config")!.Value;
        var prompt = c.Get("prompt_encoder_config")!.Value;
        int imageSize = c.GetInt("image_size") ?? 1024;
        string List(JsonElement e, string key) => string.Join(',', e.GetProperty(key).EnumerateArray().Select(v => v.GetInt32()));
        int Pair(string key) => backbone.GetProperty(key)[0].GetInt32();
        return new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "image_size"] = imageSize,
            [a + "patch.kernel"] = Pair("patch_kernel_size"),
            [a + "patch.stride"] = Pair("patch_stride"),
            [a + "patch.padding"] = Pair("patch_padding"),
            [a + "position_background"] = Pair("window_positional_embedding_background_size"),
            [a + "embed_dims"] = List(backbone, "embed_dim_per_stage"),
            [a + "blocks"] = List(backbone, "blocks_per_stage"),
            [a + "heads"] = List(backbone, "num_attention_heads_per_stage"),
            [a + "windows"] = List(backbone, "window_size_per_stage"),
            [a + "global_blocks"] = List(backbone, "global_attention_blocks"),
            [a + "query_stride"] = Pair("query_stride"),
            [a + "query_pool_stages"] = backbone.GetProperty("num_query_pool_stages").GetInt32(),
            [a + "mlp_ratio"] = (float)backbone.GetProperty("mlp_ratio").GetDouble(),
            [a + "norm_eps"] = (float)backbone.GetProperty("layer_norm_eps").GetDouble(),
            [a + "neck.size"] = vision.GetProperty("fpn_hidden_size").GetInt32(),
            [a + "neck.top_down_levels"] = List(vision, "fpn_top_down_levels"),
            [a + "decoder.head_count"] = decoder.GetProperty("num_attention_heads").GetInt32(),
            [a + "decoder.layer_count"] = decoder.GetProperty("num_hidden_layers").GetInt32(),
            [a + "decoder.downsample"] = decoder.GetProperty("attention_downsample_rate").GetInt32(),
            [a + "decoder.mask_tokens"] = decoder.GetProperty("num_multimask_outputs").GetInt32() + 1,
            [a + "prompt.patch_size"] = prompt.GetProperty("patch_size").GetInt32(),
            [CanonicalKeys.ImageProcessor] = HfImageProcessorConfig.Read(c.Directory, imageSize).ToJson().ToJsonString(),
        };
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        var collector = new HfWeightCollector();
        foreach (var (rawName, tensor) in tensors)
        {
            string name = rawName.StartsWith("model.", StringComparison.Ordinal) ? rawName["model.".Length..] : rawName;
            if (s_ignored.Any(p => name.StartsWith(p, StringComparison.Ordinal)))
            {
                continue;
            }

            if (!s_kept.Any(p => name.StartsWith(p, StringComparison.Ordinal)))
            {
                collector.AddUnmapped(rawName);
                continue;
            }

            var info = tensor.ToWeightInfo();
            if (s_dense.Any(d => name.EndsWith(d, StringComparison.Ordinal)))
            {
                collector.Add(HfTransforms.ToF32(info, name));
            }
            else
            {
                collector.Add(info.Shape.Rank <= 1 ? HfTransforms.Plain(info, name) : info.As(name));
            }
        }

        collector.ThrowIfUnmapped(c.ModelType);
        return collector.Weights;
    }
}
