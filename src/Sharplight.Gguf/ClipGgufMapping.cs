using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp vision projector files ("mmproj", GGUF architecture <c>clip</c>) → the canonical vision vocabulary documented on
/// <c>Sharplight.Vision.VisionParameters</c>. The file's <c>clip.projector_type</c> selects the tower layout and the text
/// architecture it belongs to; supported: <c>qwen3vl_merger</c> (Qwen3-VL / Qwen3.5 family ViT with 2×2 patch merger), <c>qwen2vl_merger</c>
/// (Qwen2-VL: no learned positions, quick-GELU MLP) and <c>qwen2.5vl_merger</c> (Qwen2.5-VL: RMS norms, gated SiLU MLP, windowed attention with a
/// full-attention layer every <c>clip.vision.n_wa_pattern</c> layers). A <c>qwen3vl_merger</c> tower with DeepStack layers
/// (<c>clip.vision.is_deepstack_layers</c>, Qwen3-VL) belongs to the canonical <c>decoder</c> (the Qwen3-VL language model), one without to
/// <c>qwen3_5</c>; Qwen2-VL and Qwen2.5-VL towers belong to <c>decoder</c>. Older files store separate q/k/v projections and some swap the
/// names of the MLP's up and down projections (recognised by their shapes, as llama.cpp does). <c>paddleocr</c> (PaddleOCR-VL's SigLIP tower:
/// learned position grid sized by its tensor, 2-D rotary embedding, trunk post-norm <c>v.post_ln</c>, projector <c>mm.input_norm</c> /
/// <c>mm.1</c> / <c>mm.2</c>) and <c>glm4v</c> (GLM-4.1V: patch norm, bicubic position grid, convolution merger, GLM projector) belong to
/// <c>decoder</c>.
/// The source is imported with the canonical id of the text architecture so adapters can use it as an auxiliary
/// source with role <c>mmproj</c>.
/// </summary>
internal sealed partial class ClipGgufMapping : IGgufArchitectureMapping, IGgufWeightTransformer
{
    /// <inheritdoc/>
    public IReadOnlyList<string> GgufArchitectures { get; } = ["clip"];

    /// <summary>Placeholder; the real id depends on the projector type (see <see cref="ResolveArchitecture"/>).</summary>
    public string Architecture => "clip";

    /// <summary>
    /// Tower layout of one <c>clip.projector_type</c>: the text architecture it belongs to, the processor defaults the file omits, GGUF names whose
    /// canonical name differs from the Qwen towers' (<c>Names</c>, checked first) and canonical hyper-parameters the file does not store
    /// (<c>Extra</c>, without the <c>{architecture}.vision.</c> prefix).
    /// </summary>
    private sealed record Projector(string Architecture, int TemporalPatchSize, long MinPixels, long MaxPixels, float RopeBase, string DefaultActivation = "gelu_tanh",
        string NormType = "layer", bool LearnedPositions = true, IReadOnlyDictionary<string, string>? Names = null, IReadOnlyDictionary<string, object>? Extra = null);

    // Processor defaults of the Qwen3-VL family (the file does not store them; same as its Hugging Face preprocessor_config.json).
    private static readonly Dictionary<string, Projector> s_projectors = new(StringComparer.Ordinal)
    {
        ["qwen3vl_merger"] = new("qwen3_5", TemporalPatchSize: 2, MinPixels: 65536, MaxPixels: 16777216, RopeBase: 10000f),
        // llama.cpp's default MLP activation (neither clip.use_gelu nor clip.use_silu) is the quick GELU of Qwen2-VL.
        ["qwen2vl_merger"] = new("decoder", TemporalPatchSize: 2, MinPixels: 3136, MaxPixels: 12845056, RopeBase: 10000f, DefaultActivation: "quick_gelu", LearnedPositions: false),
        ["qwen2.5vl_merger"] = new("decoder", TemporalPatchSize: 2, MinPixels: 3136, MaxPixels: 12845056, RopeBase: 10000f, DefaultActivation: "quick_gelu", NormType: "rms", LearnedPositions: false),
        // PaddleOCR-VL: v.post_ln is the trunk's final norm, the projector is mm.input_norm / mm.1 / mm.2 (LayerNorm epsilon 1e-5 in the reference).
        ["paddleocr"] = new("decoder", TemporalPatchSize: 1, MinPixels: 147456, MaxPixels: 2359296, RopeBase: 10000f,
            Names: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["v.post_ln.weight"] = "vision.post_norm.weight",
                ["v.post_ln.bias"] = "vision.post_norm.bias",
                ["mm.input_norm.weight"] = "vision.merger.norm.weight",
                ["mm.input_norm.bias"] = "vision.merger.norm.bias",
                ["mm.1.weight"] = "vision.merger.up.weight",
                ["mm.1.bias"] = "vision.merger.up.bias",
                ["mm.2.weight"] = "vision.merger.down.weight",
                ["mm.2.bias"] = "vision.merger.down.bias",
            },
            Extra: new Dictionary<string, object> { ["merger.norm_eps"] = 1e-5f, ["image_token"] = "<|IMAGE_PLACEHOLDER|>" }),
        // GLM-4.1V: RMS norm after the patch embedding (v.norm_embd), bicubic position grid, final norm v.post_ln, the 2×2 convolution
        // mm.patch_merger and the projector mm.model.fc → mm.post_norm (LayerNorm) → GELU → gated mm.gate / mm.up / mm.down.
        ["glm4v"] = new("decoder", TemporalPatchSize: 2, MinPixels: 12544, MaxPixels: 9633792, RopeBase: 10000f, DefaultActivation: "silu", NormType: "rms",
            Names: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["v.norm_embd.weight"] = "vision.patch_norm.weight",
                ["v.post_ln.weight"] = "vision.post_norm.weight",
                ["mm.patch_merger.weight"] = "vision.merger.downsample.weight",
                ["mm.patch_merger.bias"] = "vision.merger.downsample.bias",
                ["mm.model.fc.weight"] = "vision.merger.proj.weight",
                ["mm.model.fc.bias"] = "vision.merger.proj.bias",
                ["mm.post_norm.weight"] = "vision.merger.proj_norm.weight",
                ["mm.post_norm.bias"] = "vision.merger.proj_norm.bias",
                ["mm.gate.weight"] = "vision.merger.gate.weight",
                ["mm.up.weight"] = "vision.merger.up.weight",
                ["mm.down.weight"] = "vision.merger.down.weight",
            },
            Extra: new Dictionary<string, object>
            {
                ["merger.norm_eps"] = 1e-5f, ["image_token"] = "<|image|>", ["position_embedding.interpolation"] = "bicubic",
            }),
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
            // ggml's GELU is the tanh approximation.
            [a + "activation"] = g.Raw("use_silu") is true ? "silu" : g.Raw("use_gelu") is true ? "gelu_tanh" : p.DefaultActivation,
            [a + "norm.type"] = p.NormType,
            [a + "patch_size"] = patch,
            [a + "temporal_patch_size"] = p.TemporalPatchSize,
            [a + "in_channels"] = 3,
            [a + "spatial_merge_size"] = g.Int("vision.spatial_merge_size", 2),
            // The learned position grid is (image_size / patch_size)² entries.
            // The learned position grid is (image_size / patch_size)² entries; PaddleOCR files store sqrt(max_pixels) as image_size, so the
            // tensor's row count is taken when the file has it.
            [a + "position_embedding_count"] = !p.LearnedPositions ? 0
                : g.Tensor("v.position_embd.weight") is { } pos ? (int)pos.Shape[0].Fixed : (int)Math.Pow(g.Int("vision.image_size") / patch, 2),
            [a + "rope.base"] = p.RopeBase,
            [a + "out_hidden_size"] = g.Int("vision.projection_dim"),
            [a + "min_pixels"] = g.Has("vision.min_pixels") ? (long)g.Int("vision.min_pixels") : p.MinPixels,
            [a + "max_pixels"] = g.Has("vision.max_pixels") ? (long)g.Int("vision.max_pixels") : p.MaxPixels,
        };
        if (DeepstackLayers(g) is { Length: > 0 } deepstack)
        {
            md[a + "deepstack_layers"] = deepstack;
        }

        if (g.Has("vision.n_wa_pattern") && g.Int("vision.n_wa_pattern") is int pattern and > 0)
        {
            // Windows of window_size pixels (default 112, the Qwen2.5-VL value); layers il with (il + 1) % pattern == 0 attend to the whole image.
            int windowPixels = g.Has("vision.window_size") ? g.Int("vision.window_size") : 112;
            md[a + "attention.window"] = windowPixels / patch / g.Int("vision.spatial_merge_size", 2);
            md[a + "attention.full_layers"] = Enumerable.Range(0, g.Int("vision.block_count")).Where(il => (il + 1) % pattern == 0).ToArray();
        }


        if (g.Raw("vision.image_mean") is Array mean)
        {
            md[a + "image_mean"] = ToFloats(mean);
        }

        if (g.Raw("vision.image_std") is Array std)
        {
            md[a + "image_std"] = ToFloats(std);
        }

        foreach (var (key, value) in p.Extra ?? new Dictionary<string, object>())
        {
            md[a + key] = value;
        }

        // The merger activation is the trunk's (GLM-4V's gated SiLU projector); the Qwen and PaddleOCR mergers use the exact GELU.
        md[a + "merger.activation"] = p.Names?.ContainsKey("mm.gate.weight") == true ? md[a + "activation"] : "gelu";

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
        ["attn_q.weight"] = "attention.q.weight",
        ["attn_q.bias"] = "attention.q.bias",
        ["attn_k.weight"] = "attention.k.weight",
        ["attn_k.bias"] = "attention.k.bias",
        ["attn_v.weight"] = "attention.v.weight",
        ["attn_v.bias"] = "attention.v.bias",
        ["ffn_gate.weight"] = "mlp.gate.weight",
        ["ffn_gate.bias"] = "mlp.gate.bias",
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
        if (Find(g).Names is { } names && names.TryGetValue(name, out var projectorName))
        {
            return projectorName;
        }

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
        SwapLegacyMlpNames(weights, g);
        if (weights.Remove("vision.merger.downsample.weight", out var kernel) && kernel.Shape.Rank == 4)
        {
            // GLM-4V's m×m convolution kernel [out, C, m, m] as the linear map of a merged row laid out (row, column, channel).
            weights["vision.merger.downsample.weight"] = kernel.TransposeRowBlocks("vision.merger.downsample.weight", kernel.Shape[1].Fixed,
                kernel.Shape[2].Fixed * kernel.Shape[3].Fixed);
        }
        else if (kernel is not null)
        {
            weights["vision.merger.downsample.weight"] = kernel;
        }

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

    /// <summary>
    /// Files written by the old Qwen2-VL conversion scripts call the MLP's first projection <c>ffn_down</c> and its second <c>ffn_up</c>. As in
    /// llama.cpp, a "down" projection that reads hidden-size inputs is recognised as such and the two (with their biases) trade names.
    /// </summary>
    private static void SwapLegacyMlpNames(Dictionary<string, WeightInfo> weights, GgufMetadataView g)
    {
        long hidden = g.Int("vision.embedding_length");
        for (int il = 0; weights.ContainsKey($"vision.layers.{il}.mlp.down.weight"); il++)
        {
            string up = $"vision.layers.{il}.mlp.up.";
            string down = $"vision.layers.{il}.mlp.down.";
            if (!weights.TryGetValue(up + "weight", out var upWeight) || weights[down + "weight"].Shape[1].Fixed != hidden || upWeight.Shape[1].Fixed == hidden)
            {
                continue;
            }

            foreach (string kind in new[] { "weight", "bias" })
            {
                bool hasUp = weights.Remove(up + kind, out var u);
                bool hasDown = weights.Remove(down + kind, out var d);
                if (hasDown)
                {
                    weights[up + kind] = d!.As(up + kind, d.Shape);
                }

                if (hasUp)
                {
                    weights[down + kind] = u!.As(down + kind, u.Shape);
                }
            }
        }
    }
}
