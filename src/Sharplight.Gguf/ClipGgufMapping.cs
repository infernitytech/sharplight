using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp vision projector files ("mmproj", GGUF architecture <c>clip</c>) → the canonical vision vocabulary documented on
/// <c>Sharplight.Vision.VisionParameters</c>. The file's <c>clip.projector_type</c> selects the tower layout and the text
/// architecture it belongs to; supported: <c>qwen3vl_merger</c> (Qwen3-VL / Qwen3.5 family ViT with 2×2 patch merger). A tower with
/// DeepStack layers (<c>clip.vision.is_deepstack_layers</c>, Qwen3-VL) belongs to the canonical <c>decoder</c> (the Qwen3-VL language
/// model), one without to <c>qwen3_5</c>.
/// The source is imported with the canonical id of the text architecture so adapters can use it as an auxiliary
/// source with role <c>mmproj</c>.
/// </summary>
internal sealed partial class ClipGgufMapping : IGgufArchitectureMapping, IGgufWeightTransformer
{
    /// <inheritdoc/>
    public IReadOnlyList<string> GgufArchitectures { get; } = ["clip"];

    /// <summary>Placeholder; the real id depends on the projector type (see <see cref="ResolveArchitecture"/>).</summary>
    public string Architecture => "clip";

    /// <summary>Tower layout of one <c>clip.projector_type</c>: the text architecture it belongs to and the processor defaults the file omits.</summary>
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

    /// <inheritdoc/>
    public string ResolveArchitecture(GgufMetadataView g) => DeepstackLayers(g).Length > 0 ? DeepstackArchitecture : Find(g).Architecture;

    /// <summary>Text architecture of towers with DeepStack layers (Qwen3-VL: Qwen3 with M-RoPE, the generic decoder).</summary>
    private const string DeepstackArchitecture = "decoder";

    private static int[] DeepstackLayers(GgufMetadataView g) => g.Raw("vision.is_deepstack_layers") is Array flags
        ? Enumerable.Range(0, flags.Length).Where(i => Convert.ToBoolean(flags.GetValue(i))).ToArray()
        : [];

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(GgufMetadataView g)
    {
        var p = Find(g);
        string a = ResolveArchitecture(g) + ".vision.";
        int patch = g.Int("vision.patch_size");
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = g.Int("vision.block_count"),
            [a + "hidden_size"] = g.Int("vision.embedding_length"),
            [a + "ffn_hidden_size"] = g.Int("vision.feed_forward_length"),
            [a + "head_count"] = g.Int("vision.attention.head_count"),
            [a + "norm_eps"] = g.Float("vision.attention.layer_norm_epsilon", 1e-6f),
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
        if (DeepstackLayers(g) is { Length: > 0 } deepstack)
        {
            md[a + "deepstack_layers"] = deepstack;
        }

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

    [GeneratedRegex(@"^v\.deepstack\.(\d+)\.(norm|fc1|fc2)\.(weight|bias)$")]
    private static partial Regex Deepstack();

    /// <summary>Per-block tensors, GGUF suffix to canonical suffix.</summary>
    private static readonly Dictionary<string, string> s_layer = new(StringComparer.Ordinal)
    {
        ["ln1.weight"] = "input_norm.weight",
        ["ln1.bias"] = "input_norm.bias",
        ["ln2.weight"] = "post_attention_norm.weight",
        ["ln2.bias"] = "post_attention_norm.bias",
        ["attn_qkv.weight"] = "attention.qkv.weight",
        ["attn_qkv.bias"] = "attention.qkv.bias",
        ["attn_out.weight"] = "attention.o.weight",
        ["attn_out.bias"] = "attention.o.bias",
        ["ffn_up.weight"] = "mlp.up.weight",
        ["ffn_up.bias"] = "mlp.up.bias",
        ["ffn_down.weight"] = "mlp.down.weight",
        ["ffn_down.bias"] = "mlp.down.bias",
    };

    /// <summary>Tower tensors outside the blocks, GGUF name to canonical name.</summary>
    private static readonly Dictionary<string, string> s_global = new(StringComparer.Ordinal)
    {
        // One patch-embedding tensor per temporal frame, merged by Transform.
        ["v.patch_embd.weight"] = "vision.patch_embed.weight.0",
        ["v.patch_embd.weight.1"] = "vision.patch_embed.weight.1",
        ["v.patch_embd.bias"] = "vision.patch_embed.bias",
        ["v.position_embd.weight"] = "vision.position_embedding.weight",
        ["v.post_ln.weight"] = "vision.merger.norm.weight",
        ["v.post_ln.bias"] = "vision.merger.norm.bias",
        ["mm.0.weight"] = "vision.merger.up.weight",
        ["mm.0.bias"] = "vision.merger.up.bias",
        ["mm.2.weight"] = "vision.merger.down.weight",
        ["mm.2.bias"] = "vision.merger.down.bias",
    };

    /// <inheritdoc/>
    public string? MapWeightName(string name, GgufMetadataView g)
    {
        if (s_global.TryGetValue(name, out var globalName))
        {
            return globalName;
        }

        var ds = Deepstack().Match(name);
        if (ds.Success)
        {
            string module = ds.Groups[2].Value switch { "fc1" => "up", "fc2" => "down", _ => "norm" };
            return $"vision.deepstack.{ds.Groups[1].Value}.{module}.{ds.Groups[3].Value}";
        }

        var m = Block().Match(name);
        return m.Success && s_layer.TryGetValue(m.Groups[2].Value, out var c) ? $"vision.layers.{m.Groups[1].Value}.{c}" : null;
    }

    /// <summary>
    /// The patch-embedding convolution is stored as one 2-D kernel per temporal frame ([hidden, C, p, p] each);
    /// the canonical weight is the 3-D kernel flattened as [hidden, C·frames·p·p] in (channel, frame, row, col) order.
    /// </summary>
    /// <inheritdoc/>
    public void Transform(Dictionary<string, WeightInfo> weights, GgufMetadataView g)
    {
        int frames = Find(g).TemporalPatchSize;
        var parts = new WeightInfo[frames];
        for (int t = 0; t < frames; t++)
        {
            string frameName = $"vision.patch_embed.weight.{t}";
            parts[t] = weights.Remove(frameName, out var w) ? w : throw new InvalidDataException($"clip GGUF: tensor for temporal frame {t} of the patch embedding is missing.");
        }
        long hidden = parts[0].Shape[0].Fixed;
        long perFrame = parts[0].Shape.Elements() / hidden;
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
