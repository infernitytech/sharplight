using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Vision tower of CogVLM checkpoints (<c>model.vision.*</c>, EVA2-CLIP) → the canonical vision vocabulary documented on
/// <c>Sharplight.Vision.VisionParameters</c>. Images are resized to the fixed square <c>image_size</c>; the trunk prepends a learned class row
/// (<c>vision.class_embedding</c> plus position row 0, <c>vision.class_position_embedding</c>) to the patches, adds a fixed position embedding
/// (the remaining rows), and runs post-norm blocks (x + LayerNorm(attn(x)), x + LayerNorm(mlp(x))) with fused q | k | v projections with biases
/// and an exact-GELU MLP. The class row is dropped, then the GLU projector (linear, LayerNorm, GELU, SiLU-gated MLP; the canonical
/// <c>vision.merger.proj</c> / <c>proj_norm</c> / <c>gate</c> / <c>up</c> / <c>down</c>) maps every patch to the language model's width and learned
/// begin / end rows frame the image. In the language model the image rows take three positions (begin, all patches, end) and every row but the
/// end one goes through the visual experts. The prompt placeholder is <c>&lt;unk&gt;</c>, the token the remote code fills image rows with.
/// </summary>
internal static partial class CogVlmVisionHfMapping
{
    private const string Prefix = "model.vision.";

    [GeneratedRegex(@"^model\.vision\.transformer\.layers\.(\d+)\.(.+)$")]
    private static partial Regex Layer();

    /// <summary>Tensors outside the blocks → canonical names.</summary>
    private static readonly Dictionary<string, string> s_global = new(StringComparer.Ordinal)
    {
        ["model.vision.patch_embedding.proj.bias"] = "vision.patch_embed.bias",
        ["model.vision.linear_proj.linear_proj.weight"] = "vision.merger.proj.weight",
        ["model.vision.linear_proj.linear_proj.bias"] = "vision.merger.proj.bias",
        ["model.vision.linear_proj.norm1.weight"] = "vision.merger.proj_norm.weight",
        ["model.vision.linear_proj.norm1.bias"] = "vision.merger.proj_norm.bias",
        ["model.vision.linear_proj.gate_proj.weight"] = "vision.merger.gate.weight",
        ["model.vision.linear_proj.dense_h_to_4h.weight"] = "vision.merger.up.weight",
        ["model.vision.linear_proj.dense_4h_to_h.weight"] = "vision.merger.down.weight",
    };

    /// <summary>Per-layer tensors, checkpoint suffix → canonical suffix.</summary>
    private static readonly Dictionary<string, string> s_layer = new(StringComparer.Ordinal)
    {
        ["input_layernorm.weight"] = "input_norm.weight",
        ["input_layernorm.bias"] = "input_norm.bias",
        ["post_attention_layernorm.weight"] = "post_attention_norm.weight",
        ["post_attention_layernorm.bias"] = "post_attention_norm.bias",
        ["attention.query_key_value.weight"] = "attention.qkv.weight",
        ["attention.query_key_value.bias"] = "attention.qkv.bias",
        ["attention.dense.weight"] = "attention.o.weight",
        ["attention.dense.bias"] = "attention.o.bias",
        ["mlp.fc1.weight"] = "mlp.up.weight",
        ["mlp.fc1.bias"] = "mlp.up.bias",
        ["mlp.fc2.weight"] = "mlp.down.weight",
        ["mlp.fc2.bias"] = "mlp.down.bias",
    };

    /// <summary>Whether <paramref name="name"/> belongs to the tower or its projector.</summary>
    public static bool IsVisionTensor(string name) => name.StartsWith(Prefix, StringComparison.Ordinal);

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
        string act = vc.TryGetProperty("hidden_act", out var ha) ? ha.GetString() ?? "gelu" : "gelu";
        int imageSize = Int("image_size");
        md[a + "layer_count"] = Int("num_hidden_layers");
        md[a + "hidden_size"] = Int("hidden_size");
        md[a + "ffn_hidden_size"] = Int("intermediate_size");
        md[a + "head_count"] = Int("num_heads");
        md[a + "norm_eps"] = vc.TryGetProperty("layer_norm_eps", out var e) && e.ValueKind == JsonValueKind.Number ? (float)e.GetDouble() : 1e-6f;
        md[a + "activation"] = act switch
        {
            "gelu" => "gelu",
            "gelu_pytorch_tanh" or "gelu_new" => "gelu_tanh",
            _ => throw new NotSupportedException($"vision_config.hidden_act '{act}' is not supported."),
        };
        md[a + "norm.placement"] = "post";
        md[a + "class_token"] = true;
        md[a + "image_size"] = imageSize;
        md[a + "patch_size"] = Int("patch_size");
        md[a + "temporal_patch_size"] = 1;
        md[a + "in_channels"] = IntOr("in_channels", 3);
        md[a + "spatial_merge_size"] = 1;
        md[a + "position_embedding_count"] = Int("num_positions") - 1;   // without the class token's row
        md[a + "rope.base"] = 0f;
        md[a + "out_hidden_size"] = textHiddenSize;
        // The GLU projector: LayerNorm with PyTorch's default epsilon, SiLU-gated MLP.
        md[a + "merger.activation"] = "silu";
        md[a + "merger.norm_eps"] = 1e-5f;
        md[a + "merger.boundary"] = true;
        md[a + "image_positions"] = "grouped";
        md[a + "media_weight_rows"] = "all_but_last";
        md[a + "image_token"] = "<unk>";
        md[a + "image_mean"] = new[] { 0.48145466f, 0.4578275f, 0.40821073f };
        md[a + "image_std"] = new[] { 0.26862954f, 0.26130258f, 0.27577711f };
        md[a + "min_pixels"] = (long)imageSize * imageSize;
        md[a + "max_pixels"] = (long)imageSize * imageSize;
    }

    /// <summary>Maps one tower or projector tensor; null when it has no canonical counterpart.</summary>
    public static WeightInfo? MapVisionTensor(string name, SafetensorsTensor t)
    {
        var w = t.ToWeightInfo();
        switch (name)
        {
            case "model.vision.patch_embedding.proj.weight":
            {
                // The patch convolution kernel [hidden, C, p, p] flattened to the canonical [hidden, C·p·p].
                var flat = new Shape(w.Shape[0].Fixed, w.Shape.Elements() / w.Shape[0].Fixed);
                return HfTransforms.Plain(w, "vision.patch_embed.weight").As("vision.patch_embed.weight", flat);
            }
            case "model.vision.patch_embedding.position_embedding.weight":
                // Row 0 belongs to the class token (its own weight, see ClassPosition); the rest is the patch grid.
                return HfTransforms.Plain(w, "vision.position_embedding.weight").RowSlice("vision.position_embedding.weight", 1, w.Shape[0].Fixed - 1);
            case "model.vision.patch_embedding.cls_embedding":
                return Vector(w, "vision.class_embedding");
            case "model.vision.boi":
                return Vector(w, "vision.merger.begin");
            case "model.vision.eoi":
                return Vector(w, "vision.merger.end");
        }

        if (s_global.TryGetValue(name, out var globalName))
        {
            return Plain(w, globalName);
        }

        var m = Layer().Match(name);
        return m.Success && s_layer.TryGetValue(m.Groups[2].Value, out var layerName) ? Plain(w, $"vision.layers.{m.Groups[1].Value}.{layerName}") : null;
    }

    /// <summary>The class token's position row (row 0 of the position embedding) as <c>vision.class_position_embedding</c> [hidden].</summary>
    public static WeightInfo ClassPosition(SafetensorsTensor positionEmbedding)
    {
        var w = positionEmbedding.ToWeightInfo();
        return HfTransforms.ToF32(w.RowSlice("vision.class_position_embedding", 0, 1), "vision.class_position_embedding", new Shape(w.Shape[1].Fixed));
    }

    // A learned vector stored as [1, …, width] → f32 [width].
    private static WeightInfo Vector(WeightInfo w, string name) => HfTransforms.ToF32(w, name, new Shape(w.Shape[w.Shape.Rank - 1].Fixed));

    /// <summary>Norm weights, biases and vectors are held as f32; matrices stay in their stored type.</summary>
    private static WeightInfo Plain(WeightInfo w, string name) => w.Shape.Rank == 1 ? HfTransforms.ToF32(w, name) : HfTransforms.Plain(w, name);
}
