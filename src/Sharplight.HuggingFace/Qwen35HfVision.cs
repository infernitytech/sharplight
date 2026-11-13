using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.HuggingFace;

/// <summary>
/// Vision tower of Hugging Face Qwen3.5 checkpoints (<c>vision_config</c> + <c>preprocessor_config.json</c> +
/// <c>model.visual.*</c> tensors) → the canonical vision vocabulary documented on <c>Sharplight.Vision.VisionParameters</c>.
/// </summary>
public sealed partial class Qwen35HfMapping
{
    [GeneratedRegex(@"^(?:model\.)?visual\.blocks\.(\d+)\.(.+)$")]
    private static partial Regex VisualBlock();

    static JsonElement? VisionConfig(HfConfig c)
        => c.Root.TryGetProperty("vision_config", out var v) && v.ValueKind == JsonValueKind.Object ? v : null;

    static bool IsVisionTensor(string name) => name.StartsWith("model.visual.", StringComparison.Ordinal) || name.StartsWith("visual.", StringComparison.Ordinal);

    /// <summary>Adds <c>qwen3_5.vision.*</c> when the checkpoint has a vision tower.</summary>
    void AddVisionHyperParameters(HfConfig c, Dictionary<string, object> md)
    {
        if (VisionConfig(c) is not { } vc) return;
        string a = Architecture + ".vision.";
        int Int(string k) => vc.TryGetProperty(k, out var v) ? v.GetInt32() : throw new InvalidDataException($"config.json vision_config.{k} is missing.");
        int IntOr(string k, int d) => vc.TryGetProperty(k, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : d;
        if (vc.TryGetProperty("deepstack_visual_indexes", out var ds) && ds.ValueKind == JsonValueKind.Array && ds.GetArrayLength() > 0)
            throw new NotSupportedException("Vision towers with deepstack features (deepstack_visual_indexes) are not supported yet.");

        md[a + "layer_count"] = Int("depth");
        md[a + "hidden_size"] = Int("hidden_size");
        md[a + "ffn_hidden_size"] = Int("intermediate_size");
        md[a + "head_count"] = Int("num_heads");
        md[a + "norm_eps"] = 1e-6f; // fixed in the reference implementation
        string act = vc.TryGetProperty("hidden_act", out var ha) ? ha.GetString() ?? "gelu_pytorch_tanh" : "gelu_pytorch_tanh";
        md[a + "activation"] = act switch
        {
            "gelu_pytorch_tanh" or "gelu_new" or "gelu_fast" => "gelu_tanh",
            "gelu" => "gelu",
            "silu" or "swish" => "silu",
            _ => throw new NotSupportedException($"vision_config.hidden_act '{act}' is not supported."),
        };
        md[a + "patch_size"] = Int("patch_size");
        md[a + "temporal_patch_size"] = IntOr("temporal_patch_size", 1);
        md[a + "in_channels"] = IntOr("in_channels", 3);
        md[a + "spatial_merge_size"] = IntOr("spatial_merge_size", 1);
        md[a + "position_embedding_count"] = IntOr("num_position_embeddings", 0);
        md[a + "out_hidden_size"] = Int("out_hidden_size");
        md[a + "merger.activation"] = "gelu"; // the patch merger uses the exact (erf) GELU
        float theta = 10000f;
        if (vc.TryGetProperty("rope_parameters", out var rp) && rp.ValueKind == JsonValueKind.Object && rp.TryGetProperty("rope_theta", out var rt)) theta = (float)rt.GetDouble();
        md[a + "rope.base"] = theta;

        // Image preprocessing (preprocessor_config.json): normalization and pixel bounds.
        string path = Path.Combine(c.Directory, "preprocessor_config.json");
        if (!File.Exists(path)) return;
        using var doc = JsonDocument.Parse(File.ReadAllBytes(path), new JsonDocumentOptions { AllowTrailingCommas = true });
        var r = doc.RootElement;
        float[]? Floats(string k) => r.TryGetProperty(k, out var v) && v.ValueKind == JsonValueKind.Array ? v.EnumerateArray().Select(x => (float)x.GetDouble()).ToArray() : null;
        if (Floats("image_mean") is { } mean) md[a + "image_mean"] = mean;
        if (Floats("image_std") is { } std) md[a + "image_std"] = std;
        long? min = null, max = null;
        if (r.TryGetProperty("size", out var size) && size.ValueKind == JsonValueKind.Object)
        {
            if (size.TryGetProperty("shortest_edge", out var se)) min = se.GetInt64();
            if (size.TryGetProperty("longest_edge", out var le)) max = le.GetInt64();
        }
        if (r.TryGetProperty("min_pixels", out var mp)) min = mp.GetInt64();
        if (r.TryGetProperty("max_pixels", out var xp)) max = xp.GetInt64();
        if (min is { } mi) md[a + "min_pixels"] = mi;
        if (max is { } ma) md[a + "max_pixels"] = ma;
    }

    /// <summary>Maps one <c>model.visual.*</c> tensor; null when it has no canonical counterpart.</summary>
    static WeightInfo? MapVisionTensor(string rawName, SafetensorsTensor t, HfConfig c)
    {
        string name = rawName.StartsWith("model.", StringComparison.Ordinal) ? rawName["model.".Length..] : rawName;
        var w = t.ToWeightInfo();
        // Norm weights and biases (vectors) are held as f32; matrices stay in their stored type.
        WeightInfo Plain(WeightInfo x, string n) => x.Shape.Rank == 1 ? HfTransforms.ToF32(x, n) : HfTransforms.Plain(x, n);
        switch (name)
        {
            case "visual.patch_embed.proj.weight":
                return HfTransforms.Plain(w, "vision.patch_embed.weight").As("vision.patch_embed.weight", new Shape(w.Shape[0].Fixed, w.Shape.Elements() / w.Shape[0].Fixed));
            case "visual.patch_embed.proj.bias": return Plain(w, "vision.patch_embed.bias");
            case "visual.pos_embed.weight": return Plain(w, "vision.position_embedding.weight");
            case "visual.merger.norm.weight": return Plain(w, "vision.merger.norm.weight");
            case "visual.merger.norm.bias": return Plain(w, "vision.merger.norm.bias");
            case "visual.merger.linear_fc1.weight": return Plain(w, "vision.merger.up.weight");
            case "visual.merger.linear_fc1.bias": return Plain(w, "vision.merger.up.bias");
            case "visual.merger.linear_fc2.weight": return Plain(w, "vision.merger.down.weight");
            case "visual.merger.linear_fc2.bias": return Plain(w, "vision.merger.down.bias");
        }
        var m = VisualBlock().Match(rawName);
        if (!m.Success) return null;
        string p = $"vision.layers.{m.Groups[1].Value}.";
        string? canon = m.Groups[2].Value switch
        {
            "norm1.weight" => "input_norm.weight", "norm1.bias" => "input_norm.bias",
            "norm2.weight" => "post_attention_norm.weight", "norm2.bias" => "post_attention_norm.bias",
            "attn.qkv.weight" => "attention.qkv.weight", "attn.qkv.bias" => "attention.qkv.bias",
            "attn.proj.weight" => "attention.o.weight", "attn.proj.bias" => "attention.o.bias",
            "mlp.linear_fc1.weight" => "mlp.up.weight", "mlp.linear_fc1.bias" => "mlp.up.bias",
            "mlp.linear_fc2.weight" => "mlp.down.weight", "mlp.linear_fc2.bias" => "mlp.down.bias",
            _ => null,
        };
        return canon is null ? null : Plain(w, p + canon);
    }
}
