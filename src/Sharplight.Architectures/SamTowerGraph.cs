using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;
using Sharplight.Vision;

namespace Sharplight.Architectures;

/// <summary>
/// Emits the graph that encodes one fixed-size view of a two-stage image tower (<see cref="SamParameters"/>, DeepSeek-OCR): a
/// Segment-Anything ViT backbone (stride-p patch convolution, learned position grid, pre-norm blocks whose attention runs within zero-padded
/// windows or over the whole view and adds a decomposed relative position bias, exact-GELU MLP), a convolution neck (1×1 and 3×3 convolutions,
/// LayerNorms over the channels) and stride-2 convolutions; the feature map then is the patch embedding of a ViT (class token, learned position
/// grid, pre-norm, bidirectional blocks of <see cref="VisionTowerGraph.EncoderLayer"/>) whose output rows, joined with the feature map, are
/// projected to the language model's width. Every shape is static (one graph per view size); resampled position grids and relative position
/// tables are matrix products with constant interpolation weights. The pieces are generic: windowed attention is a row gather into windows
/// (padding rows gather a zero row) and back, the relative bias is two batched products of the query with gathered tables.
/// </summary>
internal static class SamTowerGraph
{
    /// <summary>Builds the graph <paramref name="name"/> for square views of <paramref name="viewPixels"/> pixels.</summary>
    public static ComputeGraph Build(VisionParameters p, Func<string, WeightInfo?> weights, string name, int viewPixels)
    {
        var sam = p.Sam ?? throw new ArgumentException("The tower has no first stage.", nameof(p));
        var b = new GraphBuilder(name, weights);
        var k = new ConstantCache(b);
        int g = viewPixels / p.PatchSize;
        int c = sam.HiddenSize;

        Value x;
        using (b.Group("sam.input"))
        {
            var pixels = b.Input(SamParameters.PixelsInput, InputRole.Media, DataType.F32, new Shape(1, 3, viewPixels, viewPixels));
            var patches = Conv(b, pixels, "vision.sam.patch_embed", p.PatchSize, 0, bias: true);                // [1, C, g, g]
            x = b.Reshape(b.Permute(patches, 0, 2, 3, 1), (long)g * g, c);
            var pos = Rows(b, k, "vision.sam.position_embedding.weight", sam.PositionGrid * sam.PositionGrid, "sam_pos");
            x = b.Add(x, ResampleGrid(b, k, pos, sam.PositionGrid, g, c), "sam_inp_pos");
        }

        for (int il = 0; il < sam.LayerCount; il++)
        {
            using var _ = b.Group($"vision.sam.layers.{il}");
            x = SamLayer(b, k, weights, sam, x, il, g);
        }

        Value features;
        int side = g;
        using (b.Group("vision.sam.neck"))
        {
            var map = b.Permute(b.Reshape(x, 1, g, g, c), 0, 3, 1, 2);                                              // [1, C, g, g]
            map = ChannelNorm(b, Conv(b, map, "vision.sam.neck.conv1", 1, 0, bias: false), "vision.sam.neck.norm1", sam.NormEps);
            map = ChannelNorm(b, Conv(b, map, "vision.sam.neck.conv2", 1, 1, bias: false), "vision.sam.neck.norm2", sam.NormEps);
            for (int j = 0; j < sam.DownsampleChannels.Length; j++)
            {
                map = Conv(b, map, $"vision.sam.downsample.{j}", 2, 1, bias: false);
                side = (side - 1) / 2 + 1;
            }

            features = b.Reshape(b.Permute(map, 0, 2, 3, 1), (long)side * side, sam.DownsampleChannels[^1]);       // [P, Cf]
        }

        int positions = side * side;
        Value h;
        using (b.Group("vision.embeddings"))
        {
            var cls = b.Add(Rows(b, k, "vision.class_embedding.weight", 1, "cls"), Rows(b, k, "vision.class_position.weight", 1, "cls_pos"));
            int grid = p.PositionGridSide;
            var pos = ResampleGrid(b, k, Rows(b, k, "vision.position_embedding.weight", grid * grid, "pos"), grid, side, p.HiddenSize);
            h = b.Concat(0, cls, b.Add(features, pos));
            h = VisionTowerGraph.Norm(b, p, h, "vision.pre_norm");
        }

        for (int il = 0; il < p.LayerCount; il++)
        {
            using var _ = b.Group($"vision.layers.{il}");
            h = VisionTowerGraph.EncoderLayer(b, p, h, il, null, null, null);
        }

        using (b.Group("projector"))
        {
            var rows = b.Concat(1, b.Slice(h, 0, 1, positions), features);
            b.Output(VisionParameters.EmbeddingsOutput, b.Linear(rows, "vision.projector.weight", "vision.projector.bias", VisionParameters.EmbeddingsOutput));
            b.Output(SamParameters.NewlineOutput, Rows(b, k, "vision.image_newline.weight", 1, "newline"));
            b.Output(SamParameters.SeparatorOutput, Rows(b, k, "vision.view_separator.weight", 1, "separator"));
        }

        return b.Build();
    }

    /// <summary>
    /// One backbone block over x [g², C]: LayerNorm, attention within windows of <see cref="SamParameters.Window"/> patches (the grid padded
    /// with zero rows to a multiple; global layers use one window over the whole grid) with the decomposed relative position bias, residual,
    /// LayerNorm, exact-GELU MLP, residual.
    /// </summary>
    private static Value SamLayer(GraphBuilder b, ConstantCache k, Func<string, WeightInfo?> weights, SamParameters sam, Value x, int il, int g)
    {
        string w = $"vision.sam.layers.{il}.";
        int c = sam.HiddenSize;
        int heads = sam.HeadCount;
        int d = sam.HeadDim;
        bool windowed = Array.IndexOf(sam.GlobalLayers, il) < 0;
        int side = windowed ? sam.Window : g;
        int perAxis = (g + side - 1) / side;
        int windows = perAxis * perAxis;
        int len = side * side;
        int batch = windows * heads;

        var h = LayerNorm(b, x, w + "input_norm", sam.NormEps);
        int[]? back = null;
        if (windowed)
        {
            // Partition: window-major rows of the padded grid; rows outside the grid gather the appended zero row.
            var (into, outOf) = WindowIndices(g, side);
            back = outOf;
            var zero = b.Scale(b.Slice(h, 0, 0, 1), 0f);
            h = b.Gather(b.Concat(0, h, zero), k.I32($"sam_win_{side}_{g}", () => into));
        }

        var qkv = b.Linear(h, w + "attention.qkv.weight", w + "attention.qkv.bias");                                  // [windows·len, 3C]
        Value Heads(int part) => b.Reshape(b.Permute(b.Reshape(b.Slice(qkv, 1, (long)part * c, c), windows, len, heads, d), 0, 2, 1, 3), batch, len, d);
        var q = Heads(0);
        var key = Heads(1);
        var v = Heads(2);
        var scores = b.Emit1("BatchMatMul", [q, key], Attributes.From(("trans_b", true), ("alpha", 1f / MathF.Sqrt(d))), $"{w}scores");
        var bias = RelativePositionBias(b, k, weights, w, q, batch, side, d);
        var probs = b.Emit1("Softmax", [b.Add(scores, bias)], null, $"{w}probs");
        var o = b.Emit1("BatchMatMul", [probs, v], null, $"{w}attn");                                                  // [batch, len, d]
        o = b.Reshape(b.Permute(b.Reshape(o, windows, heads, len, d), 0, 2, 1, 3), (long)windows * len, c);
        o = b.Linear(o, w + "attention.o.weight", w + "attention.o.bias");
        if (back is not null)
        {
            o = b.Gather(o, k.I32($"sam_unwin_{side}_{g}", () => back));
        }

        x = b.Add(x, o, $"{w}attn_residual");
        h = LayerNorm(b, x, w + "post_attention_norm", sam.NormEps);
        var up = b.Unary("Gelu", b.Linear(h, w + "mlp.up.weight", w + "mlp.up.bias"));
        return b.Add(x, b.Linear(up, w + "mlp.down.weight", w + "mlp.down.bias"), $"{w}l_out");
    }

    /// <summary>
    /// The decomposed relative position bias (MViTv2) of queries q [batch, side², d] over a square window: bias[(y, x), (ky, kx)] =
    /// q(y, x)·Rh[y − ky] + q(y, x)·Rw[x − kx], with the learned tables resampled to 2·side − 1 entries. Returns [batch, side², side²].
    /// </summary>
    private static Value RelativePositionBias(GraphBuilder b, ConstantCache k, Func<string, WeightInfo?> weights, string w, Value q, int batch, int side, int d)
    {
        var rh = RelativeTable(b, k, weights, w + "attention.rel_pos_h.weight", side, d);                                          // [side, side, d]
        var rw = RelativeTable(b, k, weights, w + "attention.rel_pos_w.weight", side, d);
        var q4 = b.Reshape(q, batch, side, side, d);
        // Height term: batch over the query row y; [y, batch·x, d] · [y, ky, d]ᵀ.
        var qh = b.Reshape(b.Permute(q4, 1, 0, 2, 3), side, (long)batch * side, d);
        var relH = b.Emit1("BatchMatMul", [qh, rh], Attributes.From(("trans_b", true)));                             // [y, batch·x, ky]
        relH = b.Reshape(b.Permute(b.Reshape(relH, side, batch, side, side), 1, 0, 2, 3), batch, (long)side * side, side, 1);
        // Width term: batch over the query column x; [x, batch·y, d] · [x, kx, d]ᵀ.
        var qw = b.Reshape(b.Permute(q4, 2, 0, 1, 3), side, (long)batch * side, d);
        var relW = b.Emit1("BatchMatMul", [qw, rw], Attributes.From(("trans_b", true)));                             // [x, batch·y, kx]
        relW = b.Reshape(b.Permute(b.Reshape(relW, side, batch, side, side), 1, 2, 0, 3), batch, (long)side * side, 1, side);
        return b.Reshape(b.Add(relH, relW), batch, (long)side * side, (long)side * side);
    }

    /// <summary>
    /// A relative position table [2·s0 − 1, d] resampled (linear) to 2·side − 1 entries and laid out per (query, key) coordinate pair:
    /// [side, side, d], entry (i, j) = table[i − j + side − 1]. One constant matrix does both.
    /// </summary>
    private static Value RelativeTable(GraphBuilder b, ConstantCache k, Func<string, WeightInfo?> weights, string weight, int side, int d)
    {
        var info = weights(weight) ?? throw new KeyNotFoundException($"Missing weight '{weight}'.");
        int source = (int)info.Shape[0].Fixed;
        int target = 2 * side - 1;
        var a = k.F32($"rel_select_{source}_{side}", new Shape((long)side * side, source), () =>
        {
            var interp = ImageResampling.Linear(source, target);
            var m = new float[(long)side * side * source];
            for (int i = 0; i < side; i++)
            {
                for (int j = 0; j < side; j++)
                {
                    int r = i - j + side - 1;
                    Array.Copy(interp, (long)r * source, m, ((long)i * side + j) * source, source);
                }
            }

            return m;
        });
        var table = Rows(b, k, weight, source, weight + ".rows");
        return b.Reshape(b.Emit1("BatchMatMul", [a, table]), side, side, d);
    }

    /// <summary>
    /// For a g × g grid in windows of side s (padded): the source row of each window-major slot (g² = the zero row past the grid) and, per grid
    /// row, its slot.
    /// </summary>
    private static (int[] Into, int[] Back) WindowIndices(int g, int s)
    {
        int perAxis = (g + s - 1) / s;
        var into = new int[perAxis * perAxis * s * s];
        var back = new int[g * g];
        int slot = 0;
        for (int wy = 0; wy < perAxis; wy++)
        {
            for (int wx = 0; wx < perAxis; wx++)
            {
                for (int y = 0; y < s; y++)
                {
                    for (int x = 0; x < s; x++, slot++)
                    {
                        int row = wy * s + y;
                        int col = wx * s + x;
                        bool inside = row < g && col < g;
                        into[slot] = inside ? row * g + col : g * g;
                        if (inside)
                        {
                            back[row * g + col] = slot;
                        }
                    }
                }
            }
        }

        return (into, back);
    }

    /// <summary>A learned grid [from², width] resampled (antialiased bicubic, separable) to [to², width]; the grid itself when the sides match.</summary>
    private static Value ResampleGrid(GraphBuilder b, ConstantCache k, Value grid, int from, int to, int width)
    {
        if (from == to)
        {
            return grid;
        }

        var wy = k.F32($"resample_{from}_{to}", new Shape(to, from), () => ImageResampling.AntialiasedBicubic(from, to));
        var wx = k.F32($"resample_{from}_{to}_batched", new Shape(1, to, from), () => ImageResampling.AntialiasedBicubic(from, to));
        var rows = b.Reshape(b.Emit1("BatchMatMul", [wy, b.Reshape(grid, from, (long)from * width)]), to, from, width);  // [to, from, W]
        return b.Reshape(b.Emit1("BatchMatMul", [wx, rows]), (long)to * to, width);                                   // [to, to, W]
    }

    /// <summary>The first <paramref name="count"/> rows of a weight table as an f32 activation.</summary>
    private static Value Rows(GraphBuilder b, ConstantCache k, string weight, int count, string name)
        => b.Gather(b.Weight(weight), k.I32($"iota_{count}", () => [.. Enumerable.Range(0, count)]), name);

    private static Value Conv(GraphBuilder b, Value x, string name, int stride, int pad, bool bias)
    {
        Value[] inputs = bias ? [b.Contiguous(x), b.Weight(name + ".weight"), b.Weight(name + ".bias")] : [b.Contiguous(x), b.Weight(name + ".weight")];
        return b.Emit1("Conv", inputs, Attributes.From(("strides", new long[] { stride, stride }), ("pads", new long[] { pad, pad, pad, pad })), name);
    }

    /// <summary>LayerNorm over the channels of a [1, C, H, W] map.</summary>
    private static Value ChannelNorm(GraphBuilder b, Value map, string name, float eps)
        => b.Permute(LayerNorm(b, b.Permute(map, 0, 2, 3, 1), name, eps), 0, 3, 1, 2);

    private static Value LayerNorm(GraphBuilder b, Value x, string name, float eps)
        => b.Emit1("LayerNorm", [x, b.Weight(name + ".weight"), b.Weight(name + ".bias")], Attributes.From(("eps", eps)));
}
