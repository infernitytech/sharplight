using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Vision tower of Hugging Face HunyuanOCR / Hunyuan-VL checkpoints (<c>vision_config</c>; Hub names <c>vit.*</c>, transformers' module names
/// <c>model.vision_tower.*</c>) → the canonical vision vocabulary documented on <c>Sharplight.Vision.VisionParameters</c>. The trunk is a ViT with
/// a learned position grid (its class row dropped) resampled bilinearly at the patch centres (align_corners off), LayerNorms, separate q/k/v
/// with biases and an exact-GELU MLP. The perceiver projector is an RMS norm per patch (<c>vision.merger.norm</c>), a stride-2 convolution over
/// each 2×2 window and a 1×1 convolution with GELU between them (<c>vision.merger.up</c> / <c>down</c>, the first kernel reordered to the merged
/// row's (row, column, channel) order), a newline column per row of merged tokens (<c>vision.merger.row_separator</c>), a linear projection to the
/// language model's width (<c>vision.merger.out</c>), begin / end rows around the image (<c>vision.merger.begin</c> / <c>end</c>) and a final
/// RMS norm (<c>vision.merger.out_norm</c>). The prompt placeholder is <c>&lt;｜hy_place▁holder▁no▁102｜&gt;</c>.
/// </summary>
internal static partial class HunyuanVlVisionHfMapping
{
    /// <summary>The image placeholder of the HunyuanOCR / Hunyuan-VL chat template.</summary>
    internal const string ImageToken = "<｜hy_place▁holder▁no▁102｜>";

    [GeneratedRegex(@"^vit\.layers\.(\d+)\.(.+)$")]
    private static partial Regex Layer();

    /// <summary>Tensors outside the blocks (Hub names) → canonical names.</summary>
    private static readonly Dictionary<string, string> s_global = new(StringComparer.Ordinal)
    {
        ["vit.embeddings.patch_embedding.bias"] = "vision.patch_embed.bias",
        ["vit.perceive.before_rms.weight"] = "vision.merger.norm.weight",
        ["vit.perceive.proj.0.bias"] = "vision.merger.up.bias",
        ["vit.perceive.proj.2.bias"] = "vision.merger.down.bias",
        ["vit.perceive.image_newline"] = "vision.merger.row_separator",
        ["vit.perceive.mlp.weight"] = "vision.merger.out.weight",
        ["vit.perceive.mlp.bias"] = "vision.merger.out.bias",
        ["vit.perceive.image_begin"] = "vision.merger.begin",
        ["vit.perceive.image_end"] = "vision.merger.end",
        ["vit.perceive.after_rms.weight"] = "vision.merger.out_norm.weight",
    };

    /// <summary>Per-layer tensors, Hub suffix → canonical suffix.</summary>
    private static readonly Dictionary<string, string> s_layer = new(StringComparer.Ordinal)
    {
        ["input_layernorm.weight"] = "input_norm.weight",
        ["input_layernorm.bias"] = "input_norm.bias",
        ["post_attention_layernorm.weight"] = "post_attention_norm.weight",
        ["post_attention_layernorm.bias"] = "post_attention_norm.bias",
        ["self_attn.q_proj.weight"] = "attention.q.weight",
        ["self_attn.q_proj.bias"] = "attention.q.bias",
        ["self_attn.k_proj.weight"] = "attention.k.weight",
        ["self_attn.k_proj.bias"] = "attention.k.bias",
        ["self_attn.v_proj.weight"] = "attention.v.weight",
        ["self_attn.v_proj.bias"] = "attention.v.bias",
        ["self_attn.o_proj.weight"] = "attention.o.weight",
        ["self_attn.o_proj.bias"] = "attention.o.bias",
        ["mlp.dense_h_to_4h.weight"] = "mlp.up.weight",
        ["mlp.dense_h_to_4h.bias"] = "mlp.up.bias",
        ["mlp.dense_4h_to_h.weight"] = "mlp.down.weight",
        ["mlp.dense_4h_to_h.bias"] = "mlp.down.bias",
    };

    /// <summary>transformers' module names → the Hub names (the reverse of its checkpoint conversion).</summary>
    private static readonly (string From, string To)[] s_moduleNames =
    [
        ("model.vision_tower.patch_merger.proj_conv.", "vit.perceive.proj.0."),
        ("model.vision_tower.patch_merger.proj_out.", "vit.perceive.proj.2."),
        ("model.vision_tower.patch_merger.", "vit.perceive."),
        ("model.vision_tower.", "vit."),
    ];

    /// <summary>Whether <paramref name="name"/> belongs to the tower or its projector.</summary>
    public static bool IsVisionTensor(string name)
        => name.StartsWith("vit.", StringComparison.Ordinal) || name.StartsWith("model.vision_tower.", StringComparison.Ordinal);

    /// <summary>Adds <c>{architecture}.vision.*</c> when the checkpoint has a <c>vision_config</c>.</summary>
    public static void AddVisionHyperParameters(HfConfig c, IDictionary<string, object> md, string architecture, int textHiddenSize)
    {
        if (QwenVisionHfMapping.VisionConfig(c) is not { } vc)
        {
            return;
        }

        string a = architecture + ".vision.";
        int Int(string k) => vc.TryGetProperty(k, out var v) ? v.GetInt32() : throw new InvalidDataException($"config.json vision_config.{k} is missing.");
        int IntOr(string k, int d) => vc.TryGetProperty(k, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : d;
        int patch = Int("patch_size");
        int side = IntOr("max_image_size", 2048) / patch;
        float eps = vc.TryGetProperty("rms_norm_eps", out var e) && e.ValueKind == JsonValueKind.Number ? (float)e.GetDouble() : 1e-5f;
        string act = vc.TryGetProperty("hidden_act", out var ha) ? ha.GetString() ?? "gelu" : "gelu";
        string activation = act switch
        {
            "gelu" => "gelu",
            "gelu_pytorch_tanh" or "gelu_new" => "gelu_tanh",
            _ => throw new NotSupportedException($"vision_config.hidden_act '{act}' is not supported."),
        };
        md[a + "layer_count"] = Int("num_hidden_layers");
        md[a + "hidden_size"] = Int("hidden_size");
        md[a + "ffn_hidden_size"] = Int("intermediate_size");
        md[a + "head_count"] = Int("num_attention_heads");
        md[a + "norm_eps"] = eps;   // the trunk's LayerNorms and the projector's RMS norms share it
        md[a + "activation"] = activation;
        md[a + "patch_size"] = patch;
        md[a + "temporal_patch_size"] = 1;
        md[a + "in_channels"] = IntOr("num_channels", 3);
        md[a + "spatial_merge_size"] = IntOr("spatial_merge_size", 2);
        md[a + "position_embedding_count"] = side * side;
        md[a + "position_embedding.interpolation"] = "bilinear_half_pixel";
        md[a + "rope.base"] = 0f;
        md[a + "out_hidden_size"] = textHiddenSize;
        md[a + "merger.activation"] = activation;
        md[a + "merger.norm.type"] = "rms";
        md[a + "merger.row_separator"] = true;
        md[a + "merger.boundary"] = true;
        md[a + "image_positions"] = "sequential";
        md[a + "image_token"] = ImageToken;
        md[a + "image_mean"] = new[] { 0.48145466f, 0.4578275f, 0.40821073f };
        md[a + "image_std"] = new[] { 0.26862954f, 0.26130258f, 0.27577711f };
        md[a + "min_pixels"] = 512L * 512;
        md[a + "max_pixels"] = 2048L * 2048;
        QwenVisionHfMapping.AddPreprocessorConfig(c, md, a);
    }

    /// <summary>Maps one tower or projector tensor; null when it has no canonical counterpart (the unused <c>image_sep</c>).</summary>
    public static WeightInfo? MapVisionTensor(string rawName, SafetensorsTensor t)
    {
        string name = HubName(rawName);
        var w = t.ToWeightInfo();
        switch (name)
        {
            case "vit.embeddings.patch_embedding.weight":
            {
                // The patch convolution kernel [hidden, C, p, p] flattened to the canonical [hidden, C·p·p].
                var flat = new Shape(w.Shape[0].Fixed, w.Shape.Elements() / w.Shape[0].Fixed);
                return HfTransforms.Plain(w, "vision.patch_embed.weight").As("vision.patch_embed.weight", flat);
            }
            case "vit.embeddings.position_embedding.weight":
                // Row 0 is the class token's, which the tower never uses.
                return HfTransforms.Plain(w, "vision.position_embedding.weight").RowSlice("vision.position_embedding.weight", 1, w.Shape[0].Fixed - 1);
            case "vit.perceive.proj.0.weight":
                // The m×m convolution kernel [out, C, m, m] as a linear map of the merged row (the window's patches in (row, column) order, C each).
                return HfTransforms.Plain(w, "vision.merger.up.weight")
                    .TransposeRowBlocks("vision.merger.up.weight", w.Shape[1].Fixed, w.Shape[2].Fixed * w.Shape[3].Fixed);
            case "vit.perceive.proj.2.weight":
            {
                // The 1×1 convolution kernel [out, in, 1, 1] is a linear map.
                var linear = new Shape(w.Shape[0].Fixed, w.Shape[1].Fixed);
                return HfTransforms.Plain(w, "vision.merger.down.weight").As("vision.merger.down.weight", linear);
            }
        }

        if (s_global.TryGetValue(name, out var globalName))
        {
            return Plain(w, globalName);
        }

        var m = Layer().Match(name);
        return m.Success && s_layer.TryGetValue(LayerSuffix(m.Groups[2].Value), out var layerName) ? Plain(w, $"vision.layers.{m.Groups[1].Value}.{layerName}") : null;
    }

    private static string HubName(string name)
    {
        foreach (var (from, to) in s_moduleNames)
        {
            if (name.StartsWith(from, StringComparison.Ordinal))
            {
                return to + name[from.Length..];
            }
        }

        return name;
    }

    // transformers' module names of a block's tensors → the Hub names.
    private static string LayerSuffix(string suffix) => suffix
        .Replace("layer_norm1.", "input_layernorm.", StringComparison.Ordinal)
        .Replace("layer_norm2.", "post_attention_layernorm.", StringComparison.Ordinal)
        .Replace("mlp.fc1.", "mlp.dense_h_to_4h.", StringComparison.Ordinal)
        .Replace("mlp.fc2.", "mlp.dense_4h_to_h.", StringComparison.Ordinal);

    /// <summary>Norm weights, biases and the projector's rows (vectors) are held as f32; matrices stay in their stored type.</summary>
    private static WeightInfo Plain(WeightInfo w, string name) => w.Shape.Rank == 1 ? HfTransforms.ToF32(w, name) : HfTransforms.Plain(w, name);
}
