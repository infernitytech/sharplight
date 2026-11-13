using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp vision projector files ("mmproj", GGUF architecture <c>clip</c>) → the canonical vision vocabulary documented on
/// <c>Sharplight.Vision.VisionParameters</c>. The file's <c>clip.projector_type</c> selects the tower layout and the text
/// architecture it belongs to; supported: <c>qwen3vl_merger</c> (Qwen3-VL / Qwen3.5 family ViT with 2×2 patch merger).
/// The source is imported with the canonical id of the text architecture so adapters can use it as an auxiliary
/// source with role <c>mmproj</c>.
/// </summary>
public sealed partial class ClipGgufMapping : IGgufArchitectureMapping, IGgufWeightTransformer
{
    public IReadOnlyList<string> GgufArchitectures { get; } = ["clip"];
    /// <summary>Placeholder; the real id depends on the projector type (see <see cref="ResolveArchitecture"/>).</summary>
    public string Architecture => "clip";

    private sealed record Projector(string Architecture, int TemporalPatchSize, long MinPixels, long MaxPixels, float RopeBase);

    // Processor defaults of the Qwen3-VL family (the file does not store them; same as its Hugging Face preprocessor_config.json).
    private static readonly Dictionary<string, Projector> s_projectors = new(StringComparer.Ordinal)
    {
        ["qwen3vl_merger"] = new("qwen3_5", TemporalPatchSize: 2, MinPixels: 65536, MaxPixels: 16777216, RopeBase: 10000f),
    };

    private static Projector Find(GgufMetadataView g)
    {
        string type = g.Raw("projector_type") as string ?? throw new InvalidDataException("clip GGUF: missing clip.projector_type.");
        return s_projectors.TryGetValue(type, out var p) ? p
            : throw new NotSupportedException($"clip GGUF projector type '{type}' is not supported (supported: {string.Join(", ", s_projectors.Keys)}).");
    }

    public string ResolveArchitecture(GgufMetadataView g) => Find(g).Architecture;

    public IReadOnlyDictionary<string, object> MapHyperParameters(GgufMetadataView g)
    {
        var p = Find(g);
        if (g.Raw("vision.is_deepstack_layers") is Array ds && ds.Cast<object>().Any(v => Convert.ToBoolean(v)))
        {
            throw new NotSupportedException("Vision towers with deepstack layers are not supported yet.");
        }

        string a = p.Architecture + ".vision.";
        int patch = g.Int("vision.patch_size");
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = g.Int("vision.block_count"),
            [a + "hidden_size"] = g.Int("vision.embedding_length"),
            [a + "ffn_hidden_size"] = g.Int("vision.feed_forward_length"),
            [a + "head_count"] = g.Int("vision.attention.head_count"),
            [a + "norm_eps"] = g.Has("vision.attention.layer_norm_epsilon") ? g.Float("vision.attention.layer_norm_epsilon") : 1e-6f,
            [a + "activation"] = g.Raw("use_silu") is true ? "silu" : "gelu_tanh", // ggml's GELU is the tanh approximation
            [a + "patch_size"] = patch,
            [a + "temporal_patch_size"] = p.TemporalPatchSize,
            [a + "in_channels"] = 3,
            [a + "spatial_merge_size"] = g.Int("vision.spatial_merge_size", 2),
            // The learned position grid is (image_size / patch_size)² entries.
            [a + "position_embedding_count"] = (int)Math.Pow(g.Int("vision.image_size") / patch, 2),
            [a + "rope.base"] = p.RopeBase,
            [a + "out_hidden_size"] = g.Int("vision.projection_dim"),
            [a + "merger.activation"] = "gelu",
            [a + "min_pixels"] = g.Has("vision.min_pixels") ? (long)g.Int("vision.min_pixels") : p.MinPixels,
            [a + "max_pixels"] = g.Has("vision.max_pixels") ? (long)g.Int("vision.max_pixels") : p.MaxPixels,
        };
        if (g.Raw("vision.image_mean") is Array mean)
        {
            md[a + "image_mean"] = ToFloats(mean);
        }

        if (g.Raw("vision.image_std") is Array std)
        {
            md[a + "image_std"] = ToFloats(std);
        }

        return md;
    }

    private static float[] ToFloats(Array a) => a.Cast<object>().Select(Convert.ToSingle).ToArray();

    [GeneratedRegex(@"^v\.blk\.(\d+)\.(.+)$")]
    private static partial Regex Block();

    private static readonly Dictionary<string, string> s_layer = new(StringComparer.Ordinal)
    {
        ["ln1.weight"] = "input_norm.weight", ["ln1.bias"] = "input_norm.bias",
        ["ln2.weight"] = "post_attention_norm.weight", ["ln2.bias"] = "post_attention_norm.bias",
        ["attn_qkv.weight"] = "attention.qkv.weight", ["attn_qkv.bias"] = "attention.qkv.bias",
        ["attn_out.weight"] = "attention.o.weight", ["attn_out.bias"] = "attention.o.bias",
        ["ffn_up.weight"] = "mlp.up.weight", ["ffn_up.bias"] = "mlp.up.bias",
        ["ffn_down.weight"] = "mlp.down.weight", ["ffn_down.bias"] = "mlp.down.bias",
    };

    private const string FrameSuffix = ".1";

    public string? MapWeightName(string name, GgufMetadataView g)
    {
        switch (name)
        {
            case "v.patch_embd.weight": return "vision.patch_embed.weight.0";   // one tensor per temporal frame, merged by Transform
            case "v.patch_embd.weight" + FrameSuffix: return "vision.patch_embed.weight.1";
            case "v.patch_embd.bias": return "vision.patch_embed.bias";
            case "v.position_embd.weight": return "vision.position_embedding.weight";
            case "v.post_ln.weight": return "vision.merger.norm.weight";
            case "v.post_ln.bias": return "vision.merger.norm.bias";
            case "mm.0.weight": return "vision.merger.up.weight";
            case "mm.0.bias": return "vision.merger.up.bias";
            case "mm.2.weight": return "vision.merger.down.weight";
            case "mm.2.bias": return "vision.merger.down.bias";
        }
        var m = Block().Match(name);
        return m.Success && s_layer.TryGetValue(m.Groups[2].Value, out var c) ? $"vision.layers.{m.Groups[1].Value}.{c}" : null;
    }

    /// <summary>
    /// The patch-embedding convolution is stored as one 2-D kernel per temporal frame ([hidden, C, p, p] each);
    /// the canonical weight is the 3-D kernel flattened as [hidden, C·frames·p·p] in (channel, frame, row, col) order.
    /// </summary>
    public void Transform(Dictionary<string, WeightInfo> weights, GgufMetadataView g)
    {
        int frames = Find(g).TemporalPatchSize;
        var parts = new WeightInfo[frames];
        for (int t = 0; t < frames; t++)
        {
            string n = $"vision.patch_embed.weight.{t}";
            parts[t] = weights.Remove(n, out var w) ? w : throw new InvalidDataException($"clip GGUF: tensor for temporal frame {t} of the patch embedding is missing.");
        }
        long hidden = parts[0].Shape[0].Fixed, perFrame = parts[0].Shape.Elements() / hidden;
        foreach (var w in parts)
        {
            if (w.Type != parts[0].Type || w.Shape.Elements() != parts[0].Shape.Elements())
            {
                throw new InvalidDataException("clip GGUF: patch embedding frames differ in type or size.");
            }
        }

        int channels = (int)(perFrame / (g.Int("vision.patch_size") * g.Int("vision.patch_size")));
        long plane = perFrame / channels;   // p·p elements per channel and frame
        var shape = new Shape(hidden, channels * frames * plane);
        var type = parts[0].Type;
        weights["vision.patch_embed.weight"] = WeightInfo.Derived("vision.patch_embed.weight", type, shape, () =>
        {
            int eb = type.ElementSize;
            var src = parts.Select(p => p.ReadAll()).ToArray();
            var dst = new byte[hidden * channels * frames * plane * eb];
            long o = 0;
            for (long h = 0; h < hidden; h++)
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int t = 0; t < frames; t++, o += plane * eb)
                    {
                        Buffer.BlockCopy(src[t], (int)((h * channels + c) * plane * eb), dst, (int)o, (int)(plane * eb));
                    }
                }
            }

            return dst;
        });
    }
}
