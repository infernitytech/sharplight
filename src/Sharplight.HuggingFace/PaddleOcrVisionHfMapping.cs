using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.HuggingFace;

/// <summary>
/// Vision tower of Hugging Face PaddleOCR-VL checkpoints (<c>vision_config</c>, <c>visual.vision_model.*</c> and the <c>mlp_AR.*</c> projector;
/// transformers 5 saves them as <c>model.visual.*</c> and <c>model.projector.*</c>) → the canonical vision vocabulary documented on
/// <c>Sharplight.Vision.VisionParameters</c>. The tower is a SigLIP ViT (learned position grid resampled bilinearly, separate q/k/v with biases,
/// tanh-GELU MLP, final <c>post_layernorm</c>) with the axial 2-D rotary embedding of the Qwen towers; the projector is a LayerNorm (epsilon
/// 1e-5) per patch, a 2×2 merge and a 2-layer GELU MLP, i.e. the canonical patch merger. The prompt placeholder is
/// <c>&lt;|IMAGE_PLACEHOLDER|&gt;</c>; image preprocessing follows the PaddleOCR-VL processor (OpenAI CLIP statistics, 384² to 1536² pixels
/// unless <c>preprocessor_config.json</c> says otherwise).
/// </summary>
internal static partial class PaddleOcrVisionHfMapping
{
    [GeneratedRegex(@"^(?:model\.)?visual\.vision_model\.encoder\.layers\.(\d+)\.(.+)$")]
    private static partial Regex EncoderLayer();

    /// <summary>Tensors outside the encoder layers (without the optional <c>model.</c> prefix) → canonical names.</summary>
    private static readonly Dictionary<string, string> s_global = new(StringComparer.Ordinal)
    {
        ["visual.vision_model.embeddings.patch_embedding.bias"] = "vision.patch_embed.bias",
        ["visual.vision_model.embeddings.position_embedding.weight"] = "vision.position_embedding.weight",
        ["visual.vision_model.post_layernorm.weight"] = "vision.post_norm.weight",
        ["visual.vision_model.post_layernorm.bias"] = "vision.post_norm.bias",
        ["mlp_AR.pre_norm.weight"] = "vision.merger.norm.weight",
        ["mlp_AR.pre_norm.bias"] = "vision.merger.norm.bias",
        ["mlp_AR.linear_1.weight"] = "vision.merger.up.weight",
        ["mlp_AR.linear_1.bias"] = "vision.merger.up.bias",
        ["mlp_AR.linear_2.weight"] = "vision.merger.down.weight",
        ["mlp_AR.linear_2.bias"] = "vision.merger.down.bias",
    };

    /// <summary>Per-layer tensors, HF suffix → canonical suffix.</summary>
    private static readonly Dictionary<string, string> s_layer = new(StringComparer.Ordinal)
    {
        ["layer_norm1.weight"] = "input_norm.weight",
        ["layer_norm1.bias"] = "input_norm.bias",
        ["layer_norm2.weight"] = "post_attention_norm.weight",
        ["layer_norm2.bias"] = "post_attention_norm.bias",
        ["self_attn.q_proj.weight"] = "attention.q.weight",
        ["self_attn.q_proj.bias"] = "attention.q.bias",
        ["self_attn.k_proj.weight"] = "attention.k.weight",
        ["self_attn.k_proj.bias"] = "attention.k.bias",
        ["self_attn.v_proj.weight"] = "attention.v.weight",
        ["self_attn.v_proj.bias"] = "attention.v.bias",
        ["self_attn.out_proj.weight"] = "attention.o.weight",
        ["self_attn.out_proj.bias"] = "attention.o.bias",
        ["mlp.fc1.weight"] = "mlp.up.weight",
        ["mlp.fc1.bias"] = "mlp.up.bias",
        ["mlp.fc2.weight"] = "mlp.down.weight",
        ["mlp.fc2.bias"] = "mlp.down.bias",
    };

    /// <summary>Whether <paramref name="name"/> belongs to the tower or its projector.</summary>
    public static bool IsVisionTensor(string name)
        => name.StartsWith("visual.", StringComparison.Ordinal) || name.StartsWith("model.visual.", StringComparison.Ordinal)
        || name.StartsWith("mlp_AR.", StringComparison.Ordinal) || name.StartsWith("model.projector.", StringComparison.Ordinal);

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
        int side = IntOr("image_size", 384) / patch;
        md[a + "layer_count"] = Int("num_hidden_layers");
        md[a + "hidden_size"] = Int("hidden_size");
        md[a + "ffn_hidden_size"] = Int("intermediate_size");
        md[a + "head_count"] = Int("num_attention_heads");
        md[a + "norm_eps"] = vc.TryGetProperty("layer_norm_eps", out var eps) ? (float)eps.GetDouble() : 1e-6f;
        string act = vc.TryGetProperty("hidden_act", out var ha) ? ha.GetString() ?? "gelu_pytorch_tanh" : "gelu_pytorch_tanh";
        md[a + "activation"] = act switch
        {
            "gelu_pytorch_tanh" or "gelu_new" or "gelu_fast" => "gelu_tanh",
            "gelu" => "gelu",
            _ => throw new NotSupportedException($"vision_config.hidden_act '{act}' is not supported."),
        };
        md[a + "patch_size"] = patch;
        md[a + "temporal_patch_size"] = 1;
        md[a + "in_channels"] = IntOr("num_channels", 3);
        md[a + "spatial_merge_size"] = IntOr("spatial_merge_size", 2);
        md[a + "position_embedding_count"] = side * side;
        md[a + "out_hidden_size"] = textHiddenSize;
        md[a + "merger.activation"] = "gelu";
        md[a + "merger.norm_eps"] = 1e-5f; // fixed in the reference projector
        float theta = 10000f;
        if (vc.TryGetProperty("rope_parameters", out var rp) && rp.ValueKind == JsonValueKind.Object && rp.TryGetProperty("rope_theta", out var rt))
        {
            theta = (float)rt.GetDouble();
        }

        md[a + "rope.base"] = theta;
        md[a + "image_token"] = "<|IMAGE_PLACEHOLDER|>";
        md[a + "image_mean"] = new[] { 0.48145466f, 0.4578275f, 0.40821073f };
        md[a + "image_std"] = new[] { 0.26862954f, 0.26130258f, 0.27577711f };
        md[a + "min_pixels"] = 384L * 384;
        md[a + "max_pixels"] = 1536L * 1536;
        QwenVisionHfMapping.AddPreprocessorConfig(c, md, a);
    }

    /// <summary>Maps one tower or projector tensor; null when it has no canonical counterpart.</summary>
    public static WeightInfo? MapVisionTensor(string rawName, SafetensorsTensor t)
    {
        string name = rawName.StartsWith("model.", StringComparison.Ordinal) ? rawName["model.".Length..] : rawName;
        if (name.StartsWith("projector.", StringComparison.Ordinal))
        {
            name = "mlp_AR." + name["projector.".Length..];
        }

        var w = t.ToWeightInfo();
        if (name == "visual.vision_model.embeddings.patch_embedding.weight")
        {
            // The patch convolution kernel [hidden, C, p, p] flattened to the canonical [hidden, C·p·p].
            var flat = new Shape(w.Shape[0].Fixed, w.Shape.Elements() / w.Shape[0].Fixed);
            return HfTransforms.Plain(w, "vision.patch_embed.weight").As("vision.patch_embed.weight", flat);
        }

        if (s_global.TryGetValue(name, out var globalName))
        {
            return Plain(w, globalName);
        }

        var m = EncoderLayer().Match(name);
        return m.Success && s_layer.TryGetValue(m.Groups[2].Value, out var layerName) ? Plain(w, $"vision.layers.{m.Groups[1].Value}.{layerName}") : null;
    }

    /// <summary>Norm weights and biases (vectors) are held as f32; matrices stay in their stored type.</summary>
    private static WeightInfo Plain(WeightInfo w, string name) => w.Shape.Rank == 1 ? HfTransforms.ToF32(w, name) : HfTransforms.Plain(w, name);
}
