using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// The image tower of DeepSeek-OCR and Unlimited-OCR (SAM ViT-B, convolution neck and downsampling, CLIP-L over the SAM feature map, linear
/// projector, newline and view-separator embeddings) → the canonical two-stage tower (<see cref="Vision.SamParameters"/>). Hyper-parameters come
/// from transformers' <c>vision_config.sam_config</c> / <c>encoder_config</c> or from the remote-code <c>vision_config.width</c>
/// (<c>sam_vit_b</c>, <c>clip-l-14-224</c>; sizes the remote code fixes are the defaults: MLP 4× the width, neck 256 channels, window 14,
/// 1024-pixel views of 16-pixel patches), view sizes and tile counts from <c>preprocessor_config.json</c> (else the reference processors'
/// defaults). Tensors keep the remote-code names in both layouts.
/// DeepSeek-OCR-2's tower (<see cref="HasQueryTower"/>) has the same SAM stage and a Qwen2 query encoder (<c>model.qwen2_model.*</c>: layers,
/// final norm, the learned queries <c>query_768</c> / <c>query_1024</c>) instead of CLIP; transformers' <c>encoder_config</c> or, in the
/// remote-code layout (<c>vision_config.width.qwen2-0-5b</c>, which only gives the width), the sizes the remote code fixes: 24 layers, 14 heads,
/// 2 key-value heads, MLP 4864, rotary base 1e6. Its views are 1024 and 768 pixels with up to 6 tiles.
/// </summary>
internal static partial class DeepseekOcrVisionHfMapping
{
    [GeneratedRegex(@"^model\.sam_model\.blocks\.(\d+)\.(.+)$")]
    private static partial Regex SamLayer();

    [GeneratedRegex(@"^model\.vision_model\.transformer\.layers\.(\d+)\.(.+)$")]
    private static partial Regex ClipLayer();

    private static readonly Dictionary<string, string> s_samLayer = new(StringComparer.Ordinal)
    {
        ["norm1.weight"] = "input_norm.weight",
        ["norm1.bias"] = "input_norm.bias",
        ["norm2.weight"] = "post_attention_norm.weight",
        ["norm2.bias"] = "post_attention_norm.bias",
        ["attn.qkv.weight"] = "attention.qkv.weight",
        ["attn.qkv.bias"] = "attention.qkv.bias",
        ["attn.proj.weight"] = "attention.o.weight",
        ["attn.proj.bias"] = "attention.o.bias",
        ["attn.rel_pos_h"] = "attention.rel_pos_h.weight",
        ["attn.rel_pos_w"] = "attention.rel_pos_w.weight",
        ["mlp.lin1.weight"] = "mlp.up.weight",
        ["mlp.lin1.bias"] = "mlp.up.bias",
        ["mlp.lin2.weight"] = "mlp.down.weight",
        ["mlp.lin2.bias"] = "mlp.down.bias",
    };

    private static readonly Dictionary<string, string> s_clipLayer = new(StringComparer.Ordinal)
    {
        ["layer_norm1.weight"] = "input_norm.weight",
        ["layer_norm1.bias"] = "input_norm.bias",
        ["layer_norm2.weight"] = "post_attention_norm.weight",
        ["layer_norm2.bias"] = "post_attention_norm.bias",
        ["self_attn.qkv_proj.weight"] = "attention.qkv.weight",
        ["self_attn.qkv_proj.bias"] = "attention.qkv.bias",
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

    [GeneratedRegex(@"^model\.qwen2_model\.model\.model\.layers\.(\d+)\.(.+)$")]
    private static partial Regex QueryLayer();

    private static readonly Dictionary<string, string> s_queryLayer = new(StringComparer.Ordinal)
    {
        ["input_layernorm.weight"] = "input_norm.weight",
        ["post_attention_layernorm.weight"] = "post_attention_norm.weight",
        ["self_attn.q_proj.weight"] = "attention.q.weight",
        ["self_attn.q_proj.bias"] = "attention.q.bias",
        ["self_attn.k_proj.weight"] = "attention.k.weight",
        ["self_attn.k_proj.bias"] = "attention.k.bias",
        ["self_attn.v_proj.weight"] = "attention.v.weight",
        ["self_attn.v_proj.bias"] = "attention.v.bias",
        ["self_attn.o_proj.weight"] = "attention.o.weight",
        ["mlp.gate_proj.weight"] = "mlp.gate.weight",
        ["mlp.up_proj.weight"] = "mlp.up.weight",
        ["mlp.down_proj.weight"] = "mlp.down.weight",
    };

    private static readonly Dictionary<string, string> s_global = new(StringComparer.Ordinal)
    {
        ["model.qwen2_model.model.model.norm.weight"] = "vision.post_norm.weight",
        ["model.sam_model.patch_embed.proj.weight"] = "vision.sam.patch_embed.weight",
        ["model.sam_model.patch_embed.proj.bias"] = "vision.sam.patch_embed.bias",
        ["model.sam_model.neck.0.weight"] = "vision.sam.neck.conv1.weight",
        ["model.sam_model.neck.1.weight"] = "vision.sam.neck.norm1.weight",
        ["model.sam_model.neck.1.bias"] = "vision.sam.neck.norm1.bias",
        ["model.sam_model.neck.2.weight"] = "vision.sam.neck.conv2.weight",
        ["model.sam_model.neck.3.weight"] = "vision.sam.neck.norm2.weight",
        ["model.sam_model.neck.3.bias"] = "vision.sam.neck.norm2.bias",
        ["model.sam_model.net_2.weight"] = "vision.sam.downsample.0.weight",
        ["model.sam_model.net_3.weight"] = "vision.sam.downsample.1.weight",
        ["model.vision_model.pre_layrnorm.weight"] = "vision.pre_norm.weight",
        ["model.vision_model.pre_layrnorm.bias"] = "vision.pre_norm.bias",
        ["model.projector.layers.weight"] = "vision.projector.weight",
        ["model.projector.layers.bias"] = "vision.projector.bias",
    };

    /// <summary>Whether the checkpoint's tower is the SAM + CLIP tower (DeepSeek-OCR, Unlimited-OCR), not DeepSeek-OCR-2's query encoder.</summary>
    public static bool HasClipTower(HfConfig c)
        => c.Root.TryGetProperty("vision_config", out var vc) && vc.ValueKind == JsonValueKind.Object
            && c.ModelType is "deepseek_vl_v2" or "unlimited_ocr"
            && !c.Architectures.Any(a => a.Contains("OCR2", StringComparison.OrdinalIgnoreCase) || a.Contains("Ocr2", StringComparison.Ordinal));

    /// <summary>Whether the checkpoint's tower is DeepSeek-OCR-2's SAM + query encoder tower.</summary>
    public static bool HasQueryTower(HfConfig c)
        => c.Root.TryGetProperty("vision_config", out var vc) && vc.ValueKind == JsonValueKind.Object
            && (c.ModelType == "deepseek_ocr2"
                || (c.ModelType == "deepseek_vl_v2" && c.Architectures.Any(a => a.Contains("OCR2", StringComparison.OrdinalIgnoreCase))));

    /// <summary>Whether the checkpoint has one of the two image towers.</summary>
    public static bool HasTower(HfConfig c) => HasClipTower(c) || HasQueryTower(c);

    /// <summary>Adds the <c>{architecture}.vision.*</c> hyper-parameters of the tower.</summary>
    public static void AddVisionHyperParameters(HfConfig c, IDictionary<string, object> md, string architecture, int textHiddenSize)
    {
        var vc = c.Root.GetProperty("vision_config");
        string a = architecture + ".vision.";
        string s = a + "sam.";
        int samWidth;
        int samPatch = 16;
        int samImage = 1024;
        bool queries = HasQueryTower(c);
        if (vc.TryGetProperty("sam_config", out var sc))
        {
            // transformers' layout.
            var ec = vc.GetProperty("encoder_config");
            samWidth = Int(sc, "hidden_size");
            samPatch = IntOr(sc, "patch_size", 16);
            samImage = IntOr(sc, "image_size", 1024);
            md[s + "layer_count"] = Int(sc, "num_hidden_layers");
            md[s + "head_count"] = Int(sc, "num_attention_heads");
            md[s + "ffn_hidden_size"] = IntOr(sc, "mlp_dim", 4 * samWidth);
            md[s + "norm_eps"] = FloatOr(sc, "layer_norm_eps", 1e-6f);
            md[s + "window"] = IntOr(sc, "window_size", 14);
            md[s + "global_layers"] = Ints(sc, "global_attn_indexes");
            md[s + "neck_channels"] = IntOr(sc, "output_channels", 256);
            md[s + "downsample_channels"] = Ints(sc, "downsample_channels");
            if (queries)
            {
                int heads = Int(ec, "num_attention_heads");
                QueryHyperParameters(md, a, Int(ec, "hidden_size"), Int(ec, "intermediate_size"), Int(ec, "num_hidden_layers"), heads,
                    IntOr(ec, "num_key_value_heads", heads), FloatOr(ec, "rms_norm_eps", 1e-6f), RopeTheta(ec, 1e6f));
            }
            else
            {
                ClipHyperParameters(md, a, Int(ec, "hidden_size"), IntOr(ec, "intermediate_size", 4 * Int(ec, "hidden_size")), Int(ec, "num_hidden_layers"),
                    Int(ec, "num_attention_heads"), IntOr(ec, "image_size", 224), IntOr(ec, "patch_size", 14), FloatOr(ec, "layer_norm_eps", 1e-5f),
                    ec.TryGetProperty("hidden_act", out var act) ? act.GetString() ?? "quick_gelu" : "quick_gelu");
            }
        }
        else
        {
            // Remote-code layout: vision_config.width.{sam_vit_b, clip-l-14-224}.
            var width = vc.GetProperty("width");
            var sam = width.GetProperty("sam_vit_b");
            samWidth = Int(sam, "width");
            samImage = IntOr(vc, "image_size", 1024);
            md[s + "layer_count"] = Int(sam, "layers");
            md[s + "head_count"] = Int(sam, "heads");
            md[s + "ffn_hidden_size"] = IntOr(sam, "mlp_dim", 4 * samWidth);
            md[s + "norm_eps"] = 1e-6f;
            md[s + "window"] = IntOr(sam, "window_size", 14);
            md[s + "global_layers"] = Ints(sam, "global_attn_indexes");
            md[s + "neck_channels"] = IntOr(sam, "output_channels", 256);
            md[s + "downsample_channels"] = sam.TryGetProperty("downsample_channels", out _) ? Ints(sam, "downsample_channels") : new[] { 512, 1024 };
            if (queries)
            {
                // The remote code builds the Qwen2 encoder from fixed sizes; the config gives its width (a tiny checkpoint may give the rest).
                var q = width.GetProperty("qwen2-0-5b");
                QueryHyperParameters(md, a, Int(q, "dim"), IntOr(q, "intermediate_size", 4864), IntOr(q, "layers", 24), IntOr(q, "heads", 14),
                    IntOr(q, "kv_heads", 2), FloatOr(q, "rms_norm_eps", 1e-6f), FloatOr(q, "rope_theta", 1e6f));
            }
            else
            {
                var clip = width.GetProperty("clip-l-14-224");
                int clipWidth = Int(clip, "width");
                ClipHyperParameters(md, a, clipWidth, IntOr(clip, "intermediate_size", 4 * clipWidth), Int(clip, "layers"), Int(clip, "heads"),
                    IntOr(clip, "image_size", 224), IntOr(clip, "patch_size", 14), 1e-5f, "quick_gelu");
            }
        }

        md[s + "hidden_size"] = samWidth;
        md[s + "position_grid"] = samImage / samPatch;
        md[a + "patch_size"] = samPatch;
        md[a + "out_hidden_size"] = textHiddenSize;
        md[a + "image_token"] = "<image>";
        md[a + "image_mean"] = new[] { 0.5f, 0.5f, 0.5f };
        md[a + "image_std"] = new[] { 0.5f, 0.5f, 0.5f };
        bool unlimited = c.ModelType == "unlimited_ocr";
        md[s + "global_size"] = 1024;
        md[s + "tile_size"] = queries ? 768 : 640;
        md[s + "min_tiles"] = 2;
        md[s + "max_tiles"] = queries ? 6 : unlimited ? 32 : 9;
        md[s + "pad_value"] = 127;
        AddPreprocessorConfig(c, md, a);
    }

    /// <summary>The canonical weights of one tower tensor (none for tensors the tower does not use).</summary>
    public static IEnumerable<WeightInfo> MapTowerTensor(string name, SafetensorsTensor t)
    {
        var w = t.ToWeightInfo();
        if (s_global.TryGetValue(name, out var canonical))
        {
            yield return HfTransforms.Plain(w, canonical);
            yield break;
        }

        switch (name)
        {
            case "model.sam_model.pos_embed":
                // [1, grid, grid, width] → [grid², width].
                yield return w.As("vision.sam.position_embedding.weight", new Tensors.Shape(w.Shape[1].Fixed * w.Shape[2].Fixed, w.Shape[3].Fixed));
                yield break;
            case "model.vision_model.embeddings.class_embedding":
                yield return w.As("vision.class_embedding.weight", new Tensors.Shape(1, w.Shape[0].Fixed));
                yield break;
            case "model.vision_model.embeddings.position_embedding.weight":
                // Row 0 is the class token's position, the rest the grid.
                yield return w.RowSlice("vision.class_position.weight", 0, 1);
                yield return w.RowSlice("vision.position_embedding.weight", 1, w.Shape[0].Fixed - 1);
                yield break;
            case "model.image_newline":
                yield return w.As("vision.image_newline.weight", new Tensors.Shape(1, w.Shape[0].Fixed));
                yield break;
            case "model.view_seperator" or "model.view_separator":
                yield return w.As("vision.view_separator.weight", new Tensors.Shape(1, w.Shape[0].Fixed));
                yield break;
        }

        var m = SamLayer().Match(name);
        if (m.Success && s_samLayer.TryGetValue(m.Groups[2].Value, out var samName))
        {
            yield return HfTransforms.Plain(w, $"vision.sam.layers.{m.Groups[1].Value}.{samName}");
            yield break;
        }

        m = ClipLayer().Match(name);
        if (m.Success && s_clipLayer.TryGetValue(m.Groups[2].Value, out var clipName))
        {
            yield return HfTransforms.Plain(w, $"vision.layers.{m.Groups[1].Value}.{clipName}");
            yield break;
        }

        m = QueryLayer().Match(name);
        if (m.Success && s_queryLayer.TryGetValue(m.Groups[2].Value, out var queryName))
        {
            yield return HfTransforms.Plain(w, $"vision.layers.{m.Groups[1].Value}.{queryName}");
            yield break;
        }

        if (name is "model.qwen2_model.query_768.weight" or "model.qwen2_model.query_1024.weight")
        {
            // One query set per feature count (144 for 768-pixel views, 256 for 1024).
            yield return HfTransforms.Plain(w, $"vision.query_embedding.{w.Shape[0].Fixed}.weight");
        }
    }

    private static void QueryHyperParameters(IDictionary<string, object> md, string a, int width, int ffn, int layers, int heads, int kvHeads, float eps, float ropeBase)
    {
        md[a + "layer_count"] = layers;
        md[a + "hidden_size"] = width;
        md[a + "ffn_hidden_size"] = ffn;
        md[a + "head_count"] = heads;
        md[a + "norm_eps"] = eps;
        md[a + "norm.type"] = "rms";
        md[a + "activation"] = "silu";
        md[a + "sam.encoder"] = "queries";
        md[a + "sam.query_kv_heads"] = kvHeads;
        md[a + "sam.query_rope_base"] = ropeBase;
    }

    private static float RopeTheta(JsonElement e, float d)
        => e.TryGetProperty("rope_parameters", out var r) && r.ValueKind == JsonValueKind.Object ? FloatOr(r, "rope_theta", d) : FloatOr(e, "rope_theta", d);

    private static void ClipHyperParameters(IDictionary<string, object> md, string a, int width, int ffn, int layers, int heads, int image, int patch, float eps, string act)
    {
        md[a + "layer_count"] = layers;
        md[a + "hidden_size"] = width;
        md[a + "ffn_hidden_size"] = ffn;
        md[a + "head_count"] = heads;
        md[a + "norm_eps"] = eps;
        md[a + "activation"] = act switch
        {
            "quick_gelu" => "quick_gelu",
            "gelu" => "gelu",
            _ => throw new NotSupportedException($"DeepSeek-OCR vision encoder activation '{act}' is not supported."),
        };
        int grid = image / patch;
        md[a + "position_embedding_count"] = grid * grid;
    }

    /// <summary>View sizes, tile counts, padding and normalization from <c>preprocessor_config.json</c>, over the defaults in <paramref name="md"/>.</summary>
    private static void AddPreprocessorConfig(HfConfig c, IDictionary<string, object> md, string a)
    {
        string path = Path.Combine(c.Directory, "preprocessor_config.json");
        if (!File.Exists(path))
        {
            return;
        }

        using var doc = JsonDocument.Parse(File.ReadAllBytes(path), new JsonDocumentOptions { AllowTrailingCommas = true });
        var r = doc.RootElement;
        string s = a + "sam.";
        if (r.TryGetProperty("size", out var size) && size.ValueKind == JsonValueKind.Object && size.TryGetProperty("height", out var h))
        {
            md[s + "global_size"] = h.GetInt32();
        }

        if (r.TryGetProperty("tile_size", out var tile) && tile.ValueKind == JsonValueKind.Number)
        {
            md[s + "tile_size"] = tile.GetInt32();
        }

        if (r.TryGetProperty("min_patches", out var min) && min.ValueKind == JsonValueKind.Number)
        {
            md[s + "min_tiles"] = min.GetInt32();
        }

        if (r.TryGetProperty("max_patches", out var max) && max.ValueKind == JsonValueKind.Number)
        {
            md[s + "max_tiles"] = max.GetInt32();
        }

        if (r.TryGetProperty("background_color", out var bg) && bg.ValueKind == JsonValueKind.Array && bg.GetArrayLength() > 0)
        {
            md[s + "pad_value"] = bg[0].GetInt32();
        }

        foreach (var key in new[] { "image_mean", "image_std" })
        {
            if (r.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.Array)
            {
                md[a + key] = v.EnumerateArray().Select(x => (float)x.GetDouble()).ToArray();
            }
        }
    }

    private static int Int(JsonElement e, string k) => e.TryGetProperty(k, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32()
        : throw new InvalidDataException($"config.json vision_config: '{k}' is missing.");

    private static int IntOr(JsonElement e, string k, int d) => e.TryGetProperty(k, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : d;

    private static float FloatOr(JsonElement e, string k, float d) => e.TryGetProperty(k, out var v) && v.ValueKind == JsonValueKind.Number ? (float)v.GetDouble() : d;

    private static int[] Ints(JsonElement e, string k) => e.TryGetProperty(k, out var v) && v.ValueKind == JsonValueKind.Array
        ? [.. v.EnumerateArray().Select(x => x.GetInt32())] : [];
}
