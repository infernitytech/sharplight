using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// The image stage of <see cref="Sam2Adapter"/> (<see cref="WellKnown.SegmentEncodeGraph"/>): the Hiera backbone over rows [h·w, C] (patch convolution,
/// bicubically resampled background position embedding plus the tiled window embedding, blocks of windowed or global multi-scale attention with max-pooled
/// queries and a projected, pooled skip at the stage changes, an exact-GELU MLP) and the feature-pyramid neck whose three finest levels become the image
/// embedding (<c>no_memory_embedding</c> added, rows [64², C]) and the two high-resolution skip maps of the mask decoder.
/// </summary>
internal static class Sam2ImageGraph
{
    private const string Backbone = "vision_encoder.backbone.";

    /// <summary>Builds the graph.</summary>
    /// <param name="src">The model source (weights and hyper-parameters).</param>
    /// <param name="p">The hyper-parameters.</param>
    public static ComputeGraph Build(ModelSource src, Sam2Parameters p)
    {
        var b = new GraphBuilder(WellKnown.SegmentEncodeGraph, src.Weight);
        var k = new ConstantCache(b);
        var pixels = b.Input("pixels", InputRole.Media, DataType.F32, new Shape(1, 3, p.ImageSize, p.ImageSize));
        int grid = (p.ImageSize + 2 * p.Patch.Padding - p.Patch.Kernel) / p.Patch.Stride + 1;
        int dim = p.EmbedDims[0];
        Value x;
        using (b.Group("embed"))
        {
            var map = Conv(b, pixels, Backbone + "patch_embed.projection", p.Patch.Stride, p.Patch.Padding);
            x = b.Reshape(b.Contiguous(b.Permute(map, 0, 2, 3, 1)), (long)grid * grid, dim);
            x = b.Add(b.Add(x, BackgroundPosition(b, k, p, grid, dim)), WindowPosition(b, k, p, grid, dim));
        }

        var stageOut = new List<(Value Rows, int Grid, int Dim)>();
        int index = 0;
        for (int stage = 0; stage < p.Blocks.Length; stage++)
        {
            for (int j = 0; j < p.Blocks[stage]; j++, index++)
            {
                using var _ = b.Group($"blocks.{index}");
                (x, grid) = Block(b, k, p, x, grid, index, stage, j);
            }

            stageOut.Add((x, grid, p.EmbedDims[stage]));
        }

        Neck(b, p, stageOut);
        return b.Build();
    }

    private static (Value X, int Grid) Block(GraphBuilder b, ConstantCache k, Sam2Parameters p, Value x, int grid, int index, int stage, int j)
    {
        string w = $"{Backbone}blocks.{index}.";
        bool first = stage > 0 && j == 0;
        int dimIn = first ? p.EmbedDims[stage - 1] : p.EmbedDims[stage];
        int dimOut = p.EmbedDims[stage];
        int window = first ? p.Windows[stage - 1] : p.Windows[stage];
        if (Array.IndexOf(p.GlobalBlocks, index) >= 0)
        {
            window = 0;
        }

        int stride = stage > 0 && stage <= p.QueryPoolStages && j == 0 ? p.QueryStride : 0;
        int heads = p.Heads[stage];
        int d = dimOut / heads;
        var h = LayerNorm(b, x, w + "layer_norm1", p.Eps);
        var residual = x;
        int outGrid = stride > 0 ? grid / stride : grid;
        if (dimIn != dimOut)
        {
            residual = b.Linear(h, w + "proj.weight", w + "proj.bias");
            if (stride > 0)
            {
                residual = PoolRows(b, residual, 1, grid, dimOut, stride);
            }
        }

        // Rows of the attention: window-major (padding rows gather a zero row) or the whole map.
        int windows = 1;
        int side = grid;
        if (window > 0)
        {
            side = window;
            windows = ((grid + window - 1) / window) * ((grid + window - 1) / window);
            var (into, _) = WindowIndices(grid, window);
            var zero = b.Scale(b.Slice(h, 0, 0, 1), 0f);
            h = b.Gather(b.Concat(0, h, zero), k.I32($"win_{window}_{grid}", () => into));
        }

        int rows = windows * side * side;
        var qkv = b.Linear(h, w + "attn.qkv.weight", w + "attn.qkv.bias");
        var q = b.Reshape(b.Slice(qkv, 1, 0, dimOut), rows, heads, d);
        var key = b.Reshape(b.Slice(qkv, 1, dimOut, dimOut), rows, heads, d);
        var v = b.Reshape(b.Slice(qkv, 1, 2L * dimOut, dimOut), rows, heads, d);
        int queryRows = rows;
        int querySide = side;
        if (stride > 0)
        {
            querySide = side / stride;
            queryRows = windows * querySide * querySide;
            q = b.Reshape(PoolRows(b, b.Reshape(q, rows, dimOut), windows, side, dimOut, stride), queryRows, heads, d);
        }

        var attrs = window > 0
            ? Attributes.From(("scale", 1f / MathF.Sqrt(d)), ("causal", false), (BatchedAttention.RowsAttribute, (long)querySide * querySide), (BatchedAttention.KeysAttribute, (long)side * side))
            : Attributes.From(("scale", 1f / MathF.Sqrt(d)), ("causal", false));
        var attn = b.Emit1("Attention", [q, key, v], attrs);
        var o = b.Linear(b.Reshape(attn, queryRows, dimOut), w + "attn.proj.weight", w + "attn.proj.bias");
        if (window > 0)
        {
            var back = WindowIndices(outGrid, querySide).Back;
            o = b.Gather(o, k.I32($"unwin_{querySide}_{outGrid}", () => back));
        }

        x = b.Add(residual, o);
        var m = LayerNorm(b, x, w + "layer_norm2", p.Eps);
        m = b.Linear(b.Unary("Gelu", b.Linear(m, w + "mlp.proj_in.weight", w + "mlp.proj_in.bias")), w + "mlp.proj_out.weight", w + "mlp.proj_out.bias");
        return (b.Add(x, m), outGrid);
    }

    private static void Neck(GraphBuilder b, Sam2Parameters p, List<(Value Rows, int Grid, int Dim)> stages)
    {
        using var _ = b.Group("neck");
        int n = stages.Count - 1;
        var levels = new Value[stages.Count];
        Value? previous = null;
        for (int i = n; i >= 0; i--)
        {
            var (rows, grid, dim) = stages[i];
            var lateral = Conv(b, ToMap(b, rows, grid, dim), $"vision_encoder.neck.convs.{n - i}", 1, 0);
            if (i == n || Array.IndexOf(p.TopDownLevels, i) < 0)
            {
                previous = lateral;
            }
            else
            {
                var side = stages[i].Grid;
                var up = b.Emit1("Resize", [previous!], Attributes.From(("size", new long[] { side, side }), ("mode", "nearest"), ("coord", "asymmetric"), ("nearest_mode", "floor")));
                previous = b.Add(lateral, up);
            }

            levels[i] = previous;
        }

        int embedGrid = stages[n - 1].Grid;
        var embedding = b.Reshape(b.Contiguous(b.Permute(b.Reshape(levels[n - 1], p.NeckSize, (long)embedGrid * embedGrid), 1, 0)), (long)embedGrid * embedGrid, p.NeckSize);
        b.Output(WellKnown.SegmentEmbeddingOutput, b.Add(embedding, b.Reshape(b.Weight("no_memory_embedding"), p.NeckSize)));
        b.Output(WellKnown.SegmentFeatures0Output, Conv(b, levels[0], "mask_decoder.conv_s0", 1, 0));
        b.Output(WellKnown.SegmentFeatures1Output, Conv(b, levels[1], "mask_decoder.conv_s1", 1, 0));
    }

    /// <summary>Rows [grid², C] as a [1, C, grid, grid] map.</summary>
    private static Value ToMap(GraphBuilder b, Value rows, int grid, int c) => b.Contiguous(b.Permute(b.Reshape(rows, 1, grid, grid, c), 0, 3, 1, 2));

    /// <summary>Max pooling (kernel = stride) of the rows of <paramref name="windows"/> square windows of <paramref name="side"/> patches each; returns the pooled rows window by window.</summary>
    private static Value PoolRows(GraphBuilder b, Value rows, int windows, int side, int c, int stride)
    {
        var map = b.Contiguous(b.Permute(b.Reshape(rows, windows, side, side, c), 0, 3, 1, 2));
        var pooled = b.Emit1("Pool", [map], Attributes.From(("mode", "max"), ("kernel", new long[] { stride, stride }), ("strides", new long[] { stride, stride })));
        int outSide = side / stride;
        return b.Reshape(b.Contiguous(b.Permute(pooled, 0, 2, 3, 1)), (long)windows * outSide * outSide, c);
    }

    /// <summary>The background position table [1, C, s, s] resampled (bicubic, PyTorch's coefficients) to the grid, as rows [grid², C].</summary>
    private static Value BackgroundPosition(GraphBuilder b, ConstantCache k, Sam2Parameters p, int grid, int dim)
    {
        int s = p.PositionBackground;
        var table = b.Contiguous(b.Permute(b.Reshape(b.Weight(Backbone + "pos_embed"), dim, s, s), 1, 2, 0));          // [s, s, C]
        var rx = k.F32($"pos_x_{s}_{grid}", new Shape(1, grid, s), () => Bicubic(s, grid));
        var ry = k.F32($"pos_y_{s}_{grid}", new Shape(grid, s), () => Bicubic(s, grid));
        var alongX = b.Emit1("BatchMatMul", [rx, table]);                                                               // [s, grid, C]
        var alongY = b.Emit1("BatchMatMul", [ry, b.Reshape(alongX, s, (long)grid * dim)]);                             // [grid, grid·C]
        return b.Reshape(alongY, (long)grid * grid, dim);
    }

    /// <summary>The window position table [1, C, w, w] tiled over the grid, as rows [grid², C].</summary>
    private static Value WindowPosition(GraphBuilder b, ConstantCache k, Sam2Parameters p, int grid, int dim)
    {
        int w = p.Windows[0];
        var table = b.Contiguous(b.Permute(b.Reshape(b.Weight(Backbone + "pos_embed_window"), dim, (long)w * w), 1, 0));    // [w², C]
        return b.Gather(table, k.I32($"pos_tile_{w}_{grid}", () =>
        {
            var indices = new int[grid * grid];
            for (int y = 0; y < grid; y++)
            {
                for (int x = 0; x < grid; x++)
                {
                    indices[y * grid + x] = (y % w) * w + (x % w);
                }
            }

            return indices;
        }));
    }

    /// <summary>The bicubic interpolation matrix [to, from] of PyTorch (<c>align_corners=False</c>, A = −0.75, edge-clamped taps).</summary>
    private static float[] Bicubic(int from, int to)
    {
        const double a = -0.75;
        double Near(double t) => ((a + 2) * t - (a + 3)) * t * t + 1;
        double Far(double t) => ((a * t - 5 * a) * t + 8 * a) * t - 4 * a;
        var m = new float[to * from];
        double scale = (double)from / to;
        for (int i = 0; i < to; i++)
        {
            double source = (i + 0.5) * scale - 0.5;
            int floor = (int)Math.Floor(source);
            double t = source - floor;
            double[] weights = [Far(t + 1), Near(t), Near(1 - t), Far(2 - t)];
            for (int tap = 0; tap < 4; tap++)
            {
                int column = Math.Clamp(floor - 1 + tap, 0, from - 1);
                m[i * from + column] += (float)weights[tap];
            }
        }

        return m;
    }

    /// <summary>
    /// For a g × g grid in windows of side s (padded): the source row of each window-major slot (g² = the zero row past the grid) and, per grid row, its slot.
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

    private static Value Conv(GraphBuilder b, Value x, string name, int stride, int pad)
        => b.Emit1("Conv", [x, b.Weight(name + ".weight"), b.Weight(name + ".bias")], Attributes.From(("strides", new long[] { stride, stride }), ("pads", new long[] { pad, pad, pad, pad })));

    private static Value LayerNorm(GraphBuilder b, Value x, string name, float eps)
        => b.Emit1("LayerNorm", [x, b.Weight(name + ".weight"), b.Weight(name + ".bias")], Attributes.From(("eps", eps)));
}
