using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Gguf;

/// <summary>
/// Projector type <c>deepseekocr</c> (DeepSeek-OCR, Unlimited-OCR): the two-stage tower of <c>Sharplight.Vision.SamParameters</c>. The SAM
/// stage is <c>v.sam.*</c> (patch convolution, position grid, blocks with relative position tables, neck <c>v.sam.neck.0</c>–<c>3</c>, stride-2
/// convolutions <c>v.sam.net_2</c> / <c>net_3</c>), the CLIP stage <c>v.class_embd</c>, <c>v.position_embd</c> (row 0 the class token's),
/// <c>v.pre_ln</c> and <c>v.blk.*</c> (its own patch embedding <c>v.patch_embd</c> is unused), the projector <c>mm.model.fc</c>, and the layout
/// rows <c>v.image_newline</c> / <c>v.view_seperator</c>. What the file does not store follows llama.cpp: 16-pixel patches, 1024-pixel global
/// views and 640-pixel tiles, SAM global-attention layers 2, 5, 8 and 11, CLIP norms at 1e-5 with the quick GELU; widths the converter writes
/// wrongly (the MLP sizes) come from the tensors. Both MLPs use the exact GELU of the reference (llama.cpp's SAM MLP the tanh approximation).
/// </summary>
internal sealed partial class ClipGgufMapping
{
    private const string DeepseekOcr = "deepseekocr";

    [GeneratedRegex(@"^v\.sam\.blk\.(\d+)\.(.+)$")]
    private static partial Regex SamBlock();

    /// <summary>SAM block tensors, GGUF suffix to canonical suffix.</summary>
    private static readonly Dictionary<string, string> s_samLayer = new(StringComparer.Ordinal)
    {
        ["pre_ln.weight"] = "input_norm.weight",
        ["pre_ln.bias"] = "input_norm.bias",
        ["post_ln.weight"] = "post_attention_norm.weight",
        ["post_ln.bias"] = "post_attention_norm.bias",
        ["attn.qkv.weight"] = "attention.qkv.weight",
        ["attn.qkv.bias"] = "attention.qkv.bias",
        ["attn.out.weight"] = "attention.o.weight",
        ["attn.out.bias"] = "attention.o.bias",
        ["attn.pos_h.weight"] = "attention.rel_pos_h.weight",
        ["attn.pos_w.weight"] = "attention.rel_pos_w.weight",
        ["mlp.lin1.weight"] = "mlp.up.weight",
        ["mlp.lin1.bias"] = "mlp.up.bias",
        ["mlp.lin2.weight"] = "mlp.down.weight",
        ["mlp.lin2.bias"] = "mlp.down.bias",
    };

    /// <summary>Tower tensors outside the blocks, GGUF name to canonical name (reshaped by <see cref="TransformDeepseekOcr"/> where needed).</summary>
    private static readonly Dictionary<string, string> s_deepseekOcr = new(StringComparer.Ordinal)
    {
        ["v.sam.patch_embd.weight"] = "vision.sam.patch_embed.weight",
        ["v.sam.patch_embd.bias"] = "vision.sam.patch_embed.bias",
        ["v.sam.pos_embd.weight"] = "vision.sam.position_embedding.weight",
        ["v.sam.neck.0.weight"] = "vision.sam.neck.conv1.weight",
        ["v.sam.neck.1.weight"] = "vision.sam.neck.norm1.weight",
        ["v.sam.neck.1.bias"] = "vision.sam.neck.norm1.bias",
        ["v.sam.neck.2.weight"] = "vision.sam.neck.conv2.weight",
        ["v.sam.neck.3.weight"] = "vision.sam.neck.norm2.weight",
        ["v.sam.neck.3.bias"] = "vision.sam.neck.norm2.bias",
        ["v.sam.net_2.weight"] = "vision.sam.downsample.0.weight",
        ["v.sam.net_3.weight"] = "vision.sam.downsample.1.weight",
        // CLIP's own patch embedding, dropped by TransformDeepseekOcr: the SAM features are the CLIP stage's patch embeddings.
        ["v.patch_embd.weight"] = "vision.patch_embed.weight",
        ["v.patch_embd.bias"] = "vision.patch_embed.bias",
        ["v.class_embd"] = "vision.class_embedding.weight",
        ["v.position_embd.weight"] = "vision.position_embedding.weight",
        ["v.pre_ln.weight"] = "vision.pre_norm.weight",
        ["v.pre_ln.bias"] = "vision.pre_norm.bias",
        ["mm.model.fc.weight"] = "vision.projector.weight",
        ["mm.model.fc.bias"] = "vision.projector.bias",
        ["v.image_newline"] = "vision.image_newline.weight",
        ["v.view_seperator"] = "vision.view_separator.weight",
    };

    private static bool IsDeepseekOcr(GgufMetadataView g) => g.Raw("projector_type") as string == DeepseekOcr;

    private static string? MapDeepseekOcrName(string name)
    {
        if (s_deepseekOcr.TryGetValue(name, out var canonical))
        {
            return canonical;
        }

        var m = SamBlock().Match(name);
        if (m.Success)
        {
            return s_samLayer.TryGetValue(m.Groups[2].Value, out var s) ? $"vision.sam.layers.{m.Groups[1].Value}.{s}" : null;
        }

        // The CLIP blocks; its patch embedding is not part of the tower.
        m = Block().Match(name);
        return m.Success && s_layer.TryGetValue(m.Groups[2].Value, out var c) ? $"vision.layers.{m.Groups[1].Value}.{c}" : null;
    }

    private static int Rows(GgufMetadataView g, string tensor)
        => (int)(g.Tensor(tensor) ?? throw new InvalidDataException($"clip GGUF: tensor {tensor} is missing.")).Shape[0].Fixed;

    private static Dictionary<string, object> DeepseekOcrHyperParameters(GgufMetadataView g, string a)
    {
        string s = a + "sam.";
        int samLayers = g.Int("vision.sam.block_count");
        var samPos = g.Tensor("v.sam.pos_embd.weight") ?? throw new InvalidDataException("clip GGUF: tensor v.sam.pos_embd.weight is missing.");
        int samPatch = (int)(g.Tensor("v.sam.patch_embd.weight") ?? throw new InvalidDataException("clip GGUF: tensor v.sam.patch_embd.weight is missing.")).Shape[2].Fixed;
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = g.Int("vision.block_count"),
            [a + "hidden_size"] = g.Int("vision.embedding_length"),
            [a + "ffn_hidden_size"] = Rows(g, "v.blk.0.ffn_up.weight"),
            [a + "head_count"] = g.Int("vision.attention.head_count"),
            // The file's epsilon is the SAM stage's; the CLIP stage uses 1e-5.
            [a + "norm_eps"] = 1e-5f,
            [a + "activation"] = "quick_gelu",
            [a + "position_embedding_count"] = Rows(g, "v.position_embd.weight") - 1,
            [a + "patch_size"] = samPatch,
            [a + "out_hidden_size"] = g.Int("vision.projection_dim"),
            [a + "image_token"] = "<image>",
            [s + "layer_count"] = samLayers,
            [s + "hidden_size"] = g.Int("vision.sam.embedding_length"),
            [s + "head_count"] = g.Int("vision.sam.head_count"),
            [s + "ffn_hidden_size"] = Rows(g, "v.sam.blk.0.mlp.lin1.weight"),
            [s + "norm_eps"] = g.Float("vision.attention.layer_norm_epsilon", 1e-6f),
            [s + "window"] = g.Int("vision.window_size", 14),
            [s + "global_layers"] = new[] { 2, 5, 8, 11 }.Where(il => il < samLayers).ToArray(),
            [s + "position_grid"] = (int)samPos.Shape[1].Fixed,
            [s + "neck_channels"] = Rows(g, "v.sam.neck.0.weight"),
            [s + "downsample_channels"] = new[] { Rows(g, "v.sam.net_2.weight"), Rows(g, "v.sam.net_3.weight") },
            [s + "global_size"] = 1024,
            [s + "tile_size"] = 640,
            [s + "min_tiles"] = g.Int("vision.preproc_min_tiles", 2),
            [s + "max_tiles"] = g.Int("vision.preproc_max_tiles", 9),
            [s + "pad_value"] = 127,
        };
        md[a + "image_mean"] = g.Raw("vision.image_mean") is Array mean ? ToFloats(mean) : new[] { 0.5f, 0.5f, 0.5f };
        md[a + "image_std"] = g.Raw("vision.image_std") is Array std ? ToFloats(std) : new[] { 0.5f, 0.5f, 0.5f };
        return md;
    }

    /// <summary>
    /// The position grids as rows, the class embedding and layout rows as [1, width], the class token's position split off the grid's; the
    /// unused CLIP patch embedding removed.
    /// </summary>
    private static void TransformDeepseekOcr(Dictionary<string, WeightInfo> weights)
    {
        weights.Remove("vision.patch_embed.weight");
        weights.Remove("vision.patch_embed.bias");
        void Reshape(string name, Func<Shape, Shape> shape)
        {
            if (weights.TryGetValue(name, out var w))
            {
                weights[name] = w.As(name, shape(w.Shape));
            }
        }

        // [1, grid, grid, width] → [grid², width].
        Reshape("vision.sam.position_embedding.weight", s => new Shape(s.Elements() / s[s.Rank - 1].Fixed, s[s.Rank - 1].Fixed));
        foreach (var name in new[] { "vision.class_embedding.weight", "vision.image_newline.weight", "vision.view_separator.weight" })
        {
            Reshape(name, s => new Shape(1, s.Elements()));
        }

        if (weights.Remove("vision.position_embedding.weight", out var pos))
        {
            weights["vision.class_position.weight"] = pos.RowSlice("vision.class_position.weight", 0, 1);
            weights["vision.position_embedding.weight"] = pos.RowSlice("vision.position_embedding.weight", 1, pos.Shape[0].Fixed - 1);
        }
    }
}
