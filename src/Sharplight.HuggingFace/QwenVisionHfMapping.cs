using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.HuggingFace;

/// <summary>
/// Vision tower of Hugging Face Qwen2-VL, Qwen2.5-VL, Qwen3-VL and Qwen3.5 checkpoints (<c>vision_config</c> + <c>preprocessor_config.json</c> +
/// <c>model.visual.*</c> tensors) → the canonical vision vocabulary documented on <c>Sharplight.Vision.VisionParameters</c>.
/// Qwen2-VL's config names the tower width <c>embed_dim</c> and the output width <c>hidden_size</c> (MLP width embed_dim·mlp_ratio, quick GELU);
/// Qwen2.5-VL uses RMS norms, a gated MLP and windowed attention (<c>window_size</c> in pixels, <c>fullatt_block_indexes</c>).
/// Qwen3-VL's DeepStack mergers (<c>deepstack_merger_list.{j}</c>, fed by tower layer <c>deepstack_visual_indexes[j]</c>) become
/// <c>vision.deepstack.{layer}.*</c>.
/// </summary>
internal static partial class QwenVisionHfMapping
{
    [GeneratedRegex(@"^(?:model\.)?visual\.blocks\.(\d+)\.(.+)$")]
    private static partial Regex VisualBlock();

    [GeneratedRegex(@"^(?:model\.)?visual\.deepstack_merger_list\.(\d+)\.(norm|linear_fc1|linear_fc2)\.(weight|bias)$")]
    private static partial Regex DeepstackMerger();

    /// <summary>The <c>vision_config</c> object, or null when the checkpoint has no vision tower.</summary>
    public static JsonElement? VisionConfig(HfConfig c)
        => c.Root.TryGetProperty("vision_config", out var v) && v.ValueKind == JsonValueKind.Object ? v : null;

    /// <summary>Whether <paramref name="name"/> is a tensor of the vision tower.</summary>
    public static bool IsVisionTensor(string name) => name.StartsWith("model.visual.", StringComparison.Ordinal) || name.StartsWith("visual.", StringComparison.Ordinal);

    /// <summary>Tower layers feeding the DeepStack mergers (<c>vision_config.deepstack_visual_indexes</c>), empty for none.</summary>
    private static int[] DeepstackLayers(JsonElement vc)
        => vc.TryGetProperty("deepstack_visual_indexes", out var ds) && ds.ValueKind == JsonValueKind.Array ? ds.EnumerateArray().Select(e => e.GetInt32()).ToArray() : [];

    /// <summary>Adds <c>{architecture}.vision.*</c> when the checkpoint has a vision tower.</summary>
    public static void AddVisionHyperParameters(HfConfig c, IDictionary<string, object> md, string architecture)
    {
        if (VisionConfig(c) is not { } vc)
        {
            return;
        }

        string a = architecture + ".vision.";
        int Int(string k) => vc.TryGetProperty(k, out var v) ? v.GetInt32() : throw new InvalidDataException($"config.json vision_config.{k} is missing.");
        int IntOr(string k, int d) => vc.TryGetProperty(k, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : d;
        if (DeepstackLayers(vc) is { Length: > 0 } deepstack)
        {
            md[a + "deepstack_layers"] = deepstack;
        }

        md[a + "layer_count"] = Int("depth");
        if (vc.TryGetProperty("embed_dim", out var embed) && embed.ValueKind == JsonValueKind.Number)
        {
            // Qwen2-VL: embed_dim is the tower width, hidden_size the language model's.
            md[a + "hidden_size"] = embed.GetInt32();
            md[a + "ffn_hidden_size"] = embed.GetInt32() * IntOr("mlp_ratio", 4);
            md[a + "out_hidden_size"] = Int("hidden_size");
        }
        else
        {
            md[a + "hidden_size"] = Int("hidden_size");
            md[a + "ffn_hidden_size"] = Int("intermediate_size");
            md[a + "out_hidden_size"] = Int("out_hidden_size");
        }

        md[a + "head_count"] = Int("num_heads");
        md[a + "norm_eps"] = 1e-6f; // fixed in the reference implementation
        bool qwen25 = (c.ModelType ?? "").StartsWith("qwen2_5_vl", StringComparison.Ordinal);
        md[a + "norm.type"] = qwen25 ? "rms" : "layer";
        // Defaults of the reference configs when vision_config omits hidden_act: quick GELU (Qwen2-VL), SiLU (Qwen2.5-VL), tanh GELU (later towers).
        string defaultAct = qwen25 ? "silu" : (c.ModelType ?? "").StartsWith("qwen2", StringComparison.Ordinal) ? "quick_gelu" : "gelu_pytorch_tanh";
        string act = vc.TryGetProperty("hidden_act", out var ha) ? ha.GetString() ?? defaultAct : defaultAct;
        md[a + "activation"] = act switch
        {
            "gelu_pytorch_tanh" or "gelu_new" or "gelu_fast" => "gelu_tanh",
            "gelu" => "gelu",
            "silu" or "swish" => "silu",
            "quick_gelu" => "quick_gelu",
            _ => throw new NotSupportedException($"vision_config.hidden_act '{act}' is not supported."),
        };
        md[a + "patch_size"] = Int("patch_size");
        md[a + "temporal_patch_size"] = IntOr("temporal_patch_size", 1);
        md[a + "in_channels"] = IntOr("in_channels", 3);
        md[a + "spatial_merge_size"] = IntOr("spatial_merge_size", 1);
        md[a + "position_embedding_count"] = IntOr("num_position_embeddings", 0);
        if (qwen25 && IntOr("window_size", 0) is int window and > 0)
        {
            md[a + "attention.window"] = window / IntOr("spatial_merge_size", 1) / Int("patch_size");
            md[a + "attention.full_layers"] = vc.TryGetProperty("fullatt_block_indexes", out var full) && full.ValueKind == JsonValueKind.Array
                ? full.EnumerateArray().Select(e => e.GetInt32()).ToArray() : Array.Empty<int>();
        }

        if ((c.ModelType ?? "").StartsWith("qwen2", StringComparison.Ordinal))
        {
            // Qwen2-VL processor defaults (OpenAI CLIP statistics), used when the checkpoint has no preprocessor_config.json.
            md[a + "image_mean"] = new[] { 0.48145466f, 0.4578275f, 0.40821073f };
            md[a + "image_std"] = new[] { 0.26862954f, 0.26130258f, 0.27577711f };
            md[a + "min_pixels"] = 3136L;
            md[a + "max_pixels"] = 12845056L;
        }

        md[a + "merger.activation"] = "gelu"; // the patch merger uses the exact (erf) GELU
        float theta = 10000f;
        if (vc.TryGetProperty("rope_parameters", out var rp) && rp.ValueKind == JsonValueKind.Object && rp.TryGetProperty("rope_theta", out var rt))
        {
            theta = (float)rt.GetDouble();
        }

        md[a + "rope.base"] = theta;
        AddPreprocessorConfig(c, md, a);
    }

    /// <summary>Image preprocessing from <c>preprocessor_config.json</c> (normalization and pixel bounds), over the defaults already in <paramref name="md"/>.</summary>
    /// <param name="c">The checkpoint's config.</param>
    /// <param name="md">Receives the entries.</param>
    /// <param name="a">The key prefix (<c>{architecture}.vision.</c>).</param>
    public static void AddPreprocessorConfig(HfConfig c, IDictionary<string, object> md, string a)
    {
        string path = Path.Combine(c.Directory, "preprocessor_config.json");
        if (!File.Exists(path))
        {
            return;
        }

        using var doc = JsonDocument.Parse(File.ReadAllBytes(path), new JsonDocumentOptions { AllowTrailingCommas = true });
        var r = doc.RootElement;
        float[]? Floats(string k) => r.TryGetProperty(k, out var v) && v.ValueKind == JsonValueKind.Array ? v.EnumerateArray().Select(x => (float)x.GetDouble()).ToArray() : null;
        if (Floats("image_mean") is { } mean)
        {
            md[a + "image_mean"] = mean;
        }

        if (Floats("image_std") is { } std)
        {
            md[a + "image_std"] = std;
        }

        long? min = null;
        long? max = null;
        if (r.TryGetProperty("size", out var size) && size.ValueKind == JsonValueKind.Object)
        {
            if (size.TryGetProperty("shortest_edge", out var se) && se.ValueKind == JsonValueKind.Number)
            {
                min = se.GetInt64();
            }

            if (size.TryGetProperty("longest_edge", out var le) && le.ValueKind == JsonValueKind.Number)
            {
                max = le.GetInt64();
            }
        }
        // transformers 5 writes unset fields as null.
        if (r.TryGetProperty("min_pixels", out var mp) && mp.ValueKind == JsonValueKind.Number)
        {
            min = mp.GetInt64();
        }

        if (r.TryGetProperty("max_pixels", out var xp) && xp.ValueKind == JsonValueKind.Number)
        {
            max = xp.GetInt64();
        }

        if (min is { } mi)
        {
            md[a + "min_pixels"] = mi;
        }

        if (max is { } ma)
        {
            md[a + "max_pixels"] = ma;
        }
    }

    /// <summary>Vision tensors outside the blocks, HF name (without the optional <c>model.</c> prefix) to canonical name.</summary>
    private static readonly Dictionary<string, string> s_visionGlobal = new(StringComparer.Ordinal)
    {
        ["visual.patch_embed.proj.bias"] = "vision.patch_embed.bias",
        ["visual.pos_embed.weight"] = "vision.position_embedding.weight",
        ["visual.merger.norm.weight"] = "vision.merger.norm.weight",
        ["visual.merger.norm.bias"] = "vision.merger.norm.bias",
        ["visual.merger.linear_fc1.weight"] = "vision.merger.up.weight",
        ["visual.merger.linear_fc1.bias"] = "vision.merger.up.bias",
        ["visual.merger.linear_fc2.weight"] = "vision.merger.down.weight",
        ["visual.merger.linear_fc2.bias"] = "vision.merger.down.bias",
        // Qwen2-VL / Qwen2.5-VL merger
        ["visual.merger.ln_q.weight"] = "vision.merger.norm.weight",
        ["visual.merger.ln_q.bias"] = "vision.merger.norm.bias",
        ["visual.merger.mlp.0.weight"] = "vision.merger.up.weight",
        ["visual.merger.mlp.0.bias"] = "vision.merger.up.bias",
        ["visual.merger.mlp.2.weight"] = "vision.merger.down.weight",
        ["visual.merger.mlp.2.bias"] = "vision.merger.down.bias",
    };

    /// <summary>Per-block vision tensors, HF suffix to canonical suffix.</summary>
    private static readonly Dictionary<string, string> s_visionLayer = new(StringComparer.Ordinal)
    {
        ["norm1.weight"] = "input_norm.weight",
        ["norm1.bias"] = "input_norm.bias",
        ["norm2.weight"] = "post_attention_norm.weight",
        ["norm2.bias"] = "post_attention_norm.bias",
        ["attn.qkv.weight"] = "attention.qkv.weight",
        ["attn.qkv.bias"] = "attention.qkv.bias",
        ["attn.proj.weight"] = "attention.o.weight",
        ["attn.proj.bias"] = "attention.o.bias",
        ["mlp.linear_fc1.weight"] = "mlp.up.weight",
        ["mlp.linear_fc1.bias"] = "mlp.up.bias",
        ["mlp.linear_fc2.weight"] = "mlp.down.weight",
        ["mlp.linear_fc2.bias"] = "mlp.down.bias",
        // Qwen2-VL
        ["mlp.fc1.weight"] = "mlp.up.weight",
        ["mlp.fc1.bias"] = "mlp.up.bias",
        ["mlp.fc2.weight"] = "mlp.down.weight",
        ["mlp.fc2.bias"] = "mlp.down.bias",
        // Qwen2.5-VL (gated)
        ["mlp.gate_proj.weight"] = "mlp.gate.weight",
        ["mlp.gate_proj.bias"] = "mlp.gate.bias",
        ["mlp.up_proj.weight"] = "mlp.up.weight",
        ["mlp.up_proj.bias"] = "mlp.up.bias",
        ["mlp.down_proj.weight"] = "mlp.down.weight",
        ["mlp.down_proj.bias"] = "mlp.down.bias",
    };

    /// <summary>Norm weights and biases (vectors) are held as f32; matrices stay in their stored type.</summary>
    private static WeightInfo PlainVision(WeightInfo w, string name)
        => w.Shape.Rank == 1 ? HfTransforms.ToF32(w, name) : HfTransforms.Plain(w, name);

    /// <summary>Maps one <c>model.visual.*</c> tensor; null when it has no canonical counterpart.</summary>
    public static WeightInfo? MapVisionTensor(HfConfig c, string rawName, SafetensorsTensor t)
    {
        string name = rawName.StartsWith("model.", StringComparison.Ordinal) ? rawName["model.".Length..] : rawName;
        var w = t.ToWeightInfo();
        if (DeepstackMerger().Match(rawName) is { Success: true } ds)
        {
            var layers = DeepstackLayers(VisionConfig(c)!.Value);
            int j = int.Parse(ds.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture);
            if (j >= layers.Length)
            {
                throw new InvalidDataException($"{rawName}: DeepStack merger {j} has no entry in vision_config.deepstack_visual_indexes.");
            }

            string module = ds.Groups[2].Value switch { "linear_fc1" => "up", "linear_fc2" => "down", _ => "norm" };
            return PlainVision(w, $"vision.deepstack.{layers[j]}.{module}.{ds.Groups[3].Value}");
        }

        if (name == "visual.patch_embed.proj.weight")
        {
            // The patch-embedding convolution kernel is flattened to [hidden, rest], the canonical 2-D form.
            var flat = new Shape(w.Shape[0].Fixed, w.Shape.Elements() / w.Shape[0].Fixed);
            return HfTransforms.Plain(w, "vision.patch_embed.weight").As("vision.patch_embed.weight", flat);
        }

        if (s_visionGlobal.TryGetValue(name, out var globalName))
        {
            return PlainVision(w, globalName);
        }

        var m = VisualBlock().Match(rawName);
        if (!m.Success || !s_visionLayer.TryGetValue(m.Groups[2].Value, out var layerName))
        {
            return null;
        }

        return PlainVision(w, $"vision.layers.{m.Groups[1].Value}.{layerName}");
    }
}
