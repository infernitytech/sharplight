using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// Vision tower of Hugging Face GLM-4.1V checkpoints (<c>glm4v</c>: <c>vision_config</c>, <c>model.visual.*</c>) → the canonical vision vocabulary
/// documented on <c>Sharplight.Vision.VisionParameters</c>. A Qwen-style ViT (fused qkv, RMS norms, gated SiLU MLP of width
/// <c>out_hidden_size</c>, axial 2-D rotary embedding) with an RMS norm after the patch embedding (<c>vision.patch_norm</c>), a learned position
/// grid resampled bicubically (grid_sample, align_corners off), a final RMS norm (<c>vision.post_norm</c>), a stride-2 convolution over each 2×2
/// window (<c>vision.merger.downsample</c>, its kernel reordered to the merged row's (row, column, channel) order) and the projector
/// linear → LayerNorm (epsilon 1e-5) → GELU → gated MLP (<c>vision.merger.proj</c>, <c>proj_norm</c>, <c>gate</c> / <c>up</c> / <c>down</c>).
/// The prompt placeholder is <c>&lt;|image|&gt;</c>.
/// </summary>
internal static class Glm4vVisionHfMapping
{
    /// <summary>Tensors outside the blocks (without the <c>model.</c> prefix) → canonical names; blocks and the patch embedding go through <see cref="QwenVisionHfMapping"/>.</summary>
    private static readonly Dictionary<string, string> s_global = new(StringComparer.Ordinal)
    {
        ["visual.post_conv_layernorm.weight"] = "vision.patch_norm.weight",
        ["visual.embeddings.position_embedding.weight"] = "vision.position_embedding.weight",
        ["visual.post_layernorm.weight"] = "vision.post_norm.weight",
        ["visual.downsample.bias"] = "vision.merger.downsample.bias",
        ["visual.merger.proj.weight"] = "vision.merger.proj.weight",
        ["visual.merger.proj.bias"] = "vision.merger.proj.bias",
        ["visual.merger.post_projection_norm.weight"] = "vision.merger.proj_norm.weight",
        ["visual.merger.post_projection_norm.bias"] = "vision.merger.proj_norm.bias",
        ["visual.merger.gate_proj.weight"] = "vision.merger.gate.weight",
        ["visual.merger.gate_proj.bias"] = "vision.merger.gate.bias",
        ["visual.merger.up_proj.weight"] = "vision.merger.up.weight",
        ["visual.merger.up_proj.bias"] = "vision.merger.up.bias",
        ["visual.merger.down_proj.weight"] = "vision.merger.down.weight",
        ["visual.merger.down_proj.bias"] = "vision.merger.down.bias",
    };

    /// <summary>Adds <c>{architecture}.vision.*</c> when the checkpoint has a <c>vision_config</c>.</summary>
    public static void AddVisionHyperParameters(HfConfig c, IDictionary<string, object> md, string architecture)
    {
        if (QwenVisionHfMapping.VisionConfig(c) is not { } vc)
        {
            return;
        }

        string a = architecture + ".vision.";
        int Int(string k) => vc.TryGetProperty(k, out var v) ? v.GetInt32() : throw new InvalidDataException($"config.json vision_config.{k} is missing.");
        int IntOr(string k, int d) => vc.TryGetProperty(k, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : d;
        int patch = Int("patch_size");
        int side = IntOr("image_size", 336) / patch;
        string act = vc.TryGetProperty("hidden_act", out var ha) ? ha.GetString() ?? "silu" : "silu";
        string activation = act switch
        {
            "silu" or "swish" => "silu",
            "gelu" => "gelu",
            "gelu_pytorch_tanh" or "gelu_new" => "gelu_tanh",
            _ => throw new NotSupportedException($"vision_config.hidden_act '{act}' is not supported."),
        };
        md[a + "layer_count"] = Int("depth");
        md[a + "hidden_size"] = Int("hidden_size");
        md[a + "ffn_hidden_size"] = Int("out_hidden_size");   // the blocks' MLP is out_hidden_size wide
        md[a + "head_count"] = Int("num_heads");
        md[a + "norm_eps"] = vc.TryGetProperty("rms_norm_eps", out var eps) ? (float)eps.GetDouble() : 1e-5f;
        md[a + "norm.type"] = "rms";
        md[a + "activation"] = activation;
        md[a + "patch_size"] = patch;
        md[a + "temporal_patch_size"] = IntOr("temporal_patch_size", 2);
        md[a + "in_channels"] = IntOr("in_channels", 3);
        md[a + "spatial_merge_size"] = IntOr("spatial_merge_size", 2);
        md[a + "position_embedding_count"] = side * side;
        md[a + "position_embedding.interpolation"] = "bicubic";
        md[a + "out_hidden_size"] = Int("out_hidden_size");
        md[a + "merger.activation"] = activation;
        md[a + "merger.norm_eps"] = 1e-5f;   // the projector's LayerNorm has PyTorch's default epsilon
        float theta = 10000f;
        if (vc.TryGetProperty("rope_parameters", out var rp) && rp.ValueKind == JsonValueKind.Object && rp.TryGetProperty("rope_theta", out var rt))
        {
            theta = (float)rt.GetDouble();
        }

        md[a + "rope.base"] = theta;
        md[a + "image_token"] = "<|image|>";
        md[a + "image_mean"] = new[] { 0.48145466f, 0.4578275f, 0.40821073f };
        md[a + "image_std"] = new[] { 0.26862954f, 0.26130258f, 0.27577711f };
        md[a + "min_pixels"] = 112L * 112;
        md[a + "max_pixels"] = 28L * 28 * 15000;
        QwenVisionHfMapping.AddPreprocessorConfig(c, md, a);
    }

    /// <summary>Maps one <c>model.visual.*</c> tensor; null when it has no canonical counterpart.</summary>
    public static WeightInfo? MapVisionTensor(HfConfig c, string rawName, SafetensorsTensor t)
    {
        string name = rawName.StartsWith("model.", StringComparison.Ordinal) ? rawName["model.".Length..] : rawName;
        var w = t.ToWeightInfo();
        if (name == "visual.downsample.weight")
        {
            // The m×m convolution kernel [out, C, m, m] as a linear map of the merged row, whose input is the window's patches in (row, column)
            // order, each C wide.
            return HfTransforms.Plain(w, "vision.merger.downsample.weight")
                .TransposeRowBlocks("vision.merger.downsample.weight", w.Shape[1].Fixed, w.Shape[2].Fixed * w.Shape[3].Fixed);
        }

        if (s_global.TryGetValue(name, out var canonical))
        {
            return w.Shape.Rank == 1 ? HfTransforms.ToF32(w, canonical) : HfTransforms.Plain(w, canonical);
        }

        return QwenVisionHfMapping.MapVisionTensor(c, rawName, t);
    }
}
