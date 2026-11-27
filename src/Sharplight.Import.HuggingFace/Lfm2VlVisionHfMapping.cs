using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Vision tower and projector of Hugging Face LFM2-VL checkpoints (<c>vision_config</c> of model type <c>siglip2_vision_model</c>,
/// <c>model.vision_tower.vision_model.*</c>, <c>model.multi_modal_projector.*</c>) → the canonical vision vocabulary documented on
/// <c>Sharplight.Vision.VisionParameters</c>, with the same keys as the GGUF <c>lfm2</c> mmproj: a SigLIP 2 NaFlex ViT (its patch embedding is a
/// linear map of (row, column, channel) patches, permuted to the canonical (channel, row, column) kernel; the learned position grid is resampled
/// to every view with antialiased bilinear interpolation), trunk post-norm, pixel unshuffle by <c>downsample_factor</c>, an optional LayerNorm
/// (epsilon 1e-5, <c>projector_use_layernorm</c>) and the projector MLP. Tiling and pixel bounds come from <c>preprocessor_config.json</c>
/// (Lfm2VlImageProcessor; config.json's copies of the same keys as fallback): <c>tile_size</c>, <c>min_tiles</c> / <c>max_tiles</c>,
/// <c>max_pixels_tolerance</c>, <c>use_thumbnail</c>, <c>do_image_splitting</c>, <c>min_image_tokens</c> / <c>max_image_tokens</c> (pixel bounds
/// in merge units), <c>resample</c>, <c>image_mean</c> / <c>image_std</c>.
/// </summary>
internal static partial class Lfm2VlVisionHfMapping
{
    private const string TowerPrefix = "model.vision_tower.vision_model.";
    private const string ProjectorPrefix = "model.multi_modal_projector.";

    [GeneratedRegex(@"^encoder\.layers\.(\d+)\.(.+)$")]
    private static partial Regex EncoderLayer();

    /// <summary>Tower tensors outside the encoder layers (below <see cref="TowerPrefix"/>) and projector tensors (below <see cref="ProjectorPrefix"/>, prefixed <c>projector.</c>).</summary>
    private static readonly Dictionary<string, string> s_global = new(StringComparer.Ordinal)
    {
        ["embeddings.patch_embedding.bias"] = "vision.patch_embed.bias",
        ["post_layernorm.weight"] = "vision.post_norm.weight",
        ["post_layernorm.bias"] = "vision.post_norm.bias",
        ["projector.layer_norm.weight"] = "vision.merger.merged_norm.weight",
        ["projector.layer_norm.bias"] = "vision.merger.merged_norm.bias",
        ["projector.linear_1.weight"] = "vision.merger.up.weight",
        ["projector.linear_1.bias"] = "vision.merger.up.bias",
        ["projector.linear_2.weight"] = "vision.merger.down.weight",
        ["projector.linear_2.bias"] = "vision.merger.down.bias",
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

    /// <summary>Image channels of the tower (<c>vision_config.num_channels</c>, default 3).</summary>
    /// <param name="c">The checkpoint's config.</param>
    public static int Channels(HfConfig c)
        => QwenVisionHfMapping.VisionConfig(c) is { } vc && vc.TryGetProperty("num_channels", out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : 3;

    /// <summary>Whether <paramref name="name"/> belongs to the tower or its projector.</summary>
    public static bool IsVisionTensor(string name)
        => name.StartsWith("model.vision_tower.", StringComparison.Ordinal) || name.StartsWith(ProjectorPrefix, StringComparison.Ordinal);

    /// <summary>Adds <c>{architecture}.vision.*</c> when the checkpoint has a <c>vision_config</c>.</summary>
    /// <param name="c">The checkpoint's config.</param>
    /// <param name="md">Receives the entries.</param>
    /// <param name="architecture">The text architecture the tower belongs to.</param>
    /// <param name="textHiddenSize">Width of the language model's embeddings.</param>
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
        int merge = c.Root.TryGetProperty("downsample_factor", out var df) && df.ValueKind == JsonValueKind.Number ? df.GetInt32() : 2;
        md[a + "layer_count"] = Int("num_hidden_layers");
        md[a + "hidden_size"] = Int("hidden_size");
        md[a + "ffn_hidden_size"] = Int("intermediate_size");
        md[a + "head_count"] = Int("num_attention_heads");
        md[a + "norm_eps"] = vc.TryGetProperty("layer_norm_eps", out var eps) ? (float)eps.GetDouble() : 1e-6f;
        md[a + "activation"] = Activation(vc.TryGetProperty("hidden_act", out var ha) ? ha.GetString() : null, "vision_config.hidden_act");
        md[a + "patch_size"] = patch;
        md[a + "temporal_patch_size"] = 1;
        md[a + "in_channels"] = IntOr("num_channels", 3);
        md[a + "spatial_merge_size"] = merge;
        md[a + "position_embedding_count"] = IntOr("num_patches", 256);
        md[a + "position_embedding.interpolation"] = "bilinear_antialias";
        md[a + "rope.base"] = 0f;
        md[a + "out_hidden_size"] = textHiddenSize;
        md[a + "merger.activation"] = Activation(c.Root.TryGetProperty("projector_hidden_act", out var pa) ? pa.GetString() : "gelu", "projector_hidden_act");
        md[a + "merger.norm_eps"] = 1e-5f;   // nn.LayerNorm's default in Lfm2VlMultiModalProjector

        // Lfm2VlImageProcessor's settings: preprocessor_config.json, else config.json's copies, else the processor's defaults.
        using var pre = ReadPreprocessorConfig(c.Directory);
        JsonElement? Setting(string k)
        {
            if (pre is not null && pre.RootElement.TryGetProperty(k, out var v) && v.ValueKind != JsonValueKind.Null)
            {
                return v;
            }

            return c.Root.TryGetProperty(k, out var r) && r.ValueKind != JsonValueKind.Null ? r : null;
        }

        int SettingInt(string k, int d) => Setting(k) is { ValueKind: JsonValueKind.Number } v ? v.GetInt32() : d;
        bool SettingBool(string k, bool d) => Setting(k) is { } v && v.ValueKind is JsonValueKind.True or JsonValueKind.False ? v.GetBoolean() : d;
        float[]? SettingFloats(string k) => Setting(k) is { ValueKind: JsonValueKind.Array } v ? v.EnumerateArray().Select(x => (float)x.GetDouble()).ToArray() : null;

        long unit = (long)patch * patch * merge * merge;
        md[a + "tile_size"] = SettingInt("tile_size", 512);
        md[a + "tile.min_tiles"] = SettingInt("min_tiles", 2);
        // Without image splitting the processor only smart-resizes (the preprocessor tiles only for max_tiles > 1).
        md[a + "tile.max_tiles"] = SettingBool("do_image_splitting", true) ? SettingInt("max_tiles", 10) : 1;
        md[a + "tile.max_pixels_tolerance"] = Setting("max_pixels_tolerance") is { ValueKind: JsonValueKind.Number } tol ? (float)tol.GetDouble() : 2f;
        md[a + "tile.use_thumbnail"] = SettingBool("use_thumbnail", true);
        md[a + "min_pixels"] = SettingInt("min_image_tokens", 64) * unit;
        md[a + "max_pixels"] = SettingInt("max_image_tokens", 256) * unit;
        // PIL resampling constants: 0 nearest, 1 lanczos, 2 bilinear, 3 bicubic (the processor's default).
        md[a + "resize.filter"] = SettingInt("resample", 3) switch
        {
            0 => "nearest",
            1 => "lanczos",
            2 => "bilinear",
            3 => "bicubic",
            int other => throw new NotSupportedException($"lfm2_vl: resample {other} is not supported."),
        };
        md[a + "image_mean"] = SettingFloats("image_mean") ?? [0.5f, 0.5f, 0.5f];
        md[a + "image_std"] = SettingFloats("image_std") ?? [0.5f, 0.5f, 0.5f];

        // Lfm2VlProcessor's prompt layout; checkpoints without the image special tokens use a different layout.
        bool specialTokens = ReadProcessorFlag(c.Directory, "use_image_special_tokens")
            ?? !(c.Root.TryGetProperty("use_image_special_tokens", out var st) && st.ValueKind == JsonValueKind.False);
        if (!specialTokens)
        {
            throw new NotSupportedException("lfm2_vl: use_image_special_tokens = false is not supported.");
        }

        md[a + "image_token"] = "<image>";
        md[a + "image_begin_token"] = "<|image_start|>";
        md[a + "image_end_token"] = "<|image_end|>";
        md[a + "tile.layout"] = "lfm2";
        md[a + "tile.global_token"] = "<|img_thumbnail|>";
    }

    /// <summary>Maps one tower or projector tensor; null when it has no canonical counterpart.</summary>
    /// <param name="rawName">The checkpoint's tensor name.</param>
    /// <param name="t">The tensor.</param>
    /// <param name="channels">Image channels (<c>vision_config.num_channels</c>).</param>
    public static WeightInfo? MapVisionTensor(string rawName, SafetensorsTensor t, int channels)
    {
        string name = rawName.StartsWith(TowerPrefix, StringComparison.Ordinal) ? rawName[TowerPrefix.Length..]
            : rawName.StartsWith(ProjectorPrefix, StringComparison.Ordinal) ? "projector." + rawName[ProjectorPrefix.Length..]
            : rawName;
        var w = t.ToWeightInfo();
        if (name == "embeddings.patch_embedding.weight")
        {
            // nn.Linear [hidden, p·p·C] over (row, column, channel) patches → the canonical kernel [hidden, C·p·p] in (channel, row, column) order,
            // like llama.cpp's converter (reshape to [hidden, p, p, C], permute to [hidden, C, p, p]).
            long pixels = w.Shape[1].Fixed / channels;
            return w.TransposeRowBlocks("vision.patch_embed.weight", pixels, channels);
        }

        if (name == "embeddings.position_embedding.weight")
        {
            // The antialiased resample is a dense product with the table, which the tower graph runs as an f32 activation.
            return HfTransforms.ToF32(w, "vision.position_embedding.weight");
        }

        if (s_global.TryGetValue(name, out var globalName))
        {
            return Plain(w, globalName);
        }

        var m = EncoderLayer().Match(name);
        return m.Success && s_layer.TryGetValue(m.Groups[2].Value, out var layerName) ? Plain(w, $"vision.layers.{m.Groups[1].Value}.{layerName}") : null;
    }

    /// <summary>The tower's unused pooling head (<c>vision_use_head</c>), which LFM2-VL never runs.</summary>
    /// <param name="rawName">The checkpoint's tensor name.</param>
    public static bool IsUnusedTensor(string rawName) => rawName.StartsWith(TowerPrefix + "head.", StringComparison.Ordinal);

    private static string Activation(string? act, string key) => act switch
    {
        "gelu_pytorch_tanh" or "gelu_new" or "gelu_fast" => "gelu_tanh",
        "gelu" or null => "gelu",
        _ => throw new NotSupportedException($"lfm2_vl: {key} '{act}' is not supported."),
    };

    private static JsonDocument? ReadPreprocessorConfig(string directory)
    {
        string path = Path.Combine(directory, "preprocessor_config.json");
        return File.Exists(path) ? JsonDocument.Parse(File.ReadAllBytes(path), new JsonDocumentOptions { AllowTrailingCommas = true }) : null;
    }

    private static bool? ReadProcessorFlag(string directory, string key)
    {
        string path = Path.Combine(directory, "processor_config.json");
        if (!File.Exists(path))
        {
            return null;
        }

        using var doc = JsonDocument.Parse(File.ReadAllBytes(path), new JsonDocumentOptions { AllowTrailingCommas = true });
        return doc.RootElement.TryGetProperty(key, out var v) && v.ValueKind is JsonValueKind.True or JsonValueKind.False ? v.GetBoolean() : null;
    }

    /// <summary>Norm weights and biases (vectors) are held as f32; matrices stay in their stored type.</summary>
    private static WeightInfo Plain(WeightInfo w, string name) => w.Shape.Rank == 1 ? HfTransforms.ToF32(w, name) : HfTransforms.Plain(w, name);
}
