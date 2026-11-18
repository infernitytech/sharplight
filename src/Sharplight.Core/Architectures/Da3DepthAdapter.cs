using System.Text.Json;
using System.Text.Json.Nodes;
using Sharplight.Graph;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Depth Anything 3 (canonical arch <c>da3_depth</c>) as a single-image depth estimator: a DINOv2 backbone whose later layers alternate between
/// local attention and (for one view, identical) global attention with 2-D rotary positions, per-head query / key layer norms and a camera token
/// in place of the class token, four tapped layers whose local and final features are concatenated, and the main branch of the DualDPT head
/// (reassemble, feature fusion, output convolutions with a UV position embedding; the depth is the exponential of the first output channel).
/// Builds <see cref="WellKnown.DepthGraph"/> for a fixed square input and describes the model as <see cref="ModelKinds.DepthEstimation"/>.
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>da3_depth.*</c>): <c>patch_size</c>, <c>image_size</c>, <c>backbone.{layer_count,hidden_size,head_count,alt_start,qknorm_start,rope_start,
/// rope_frequency,out_layers,cat_token}</c> (<c>out_layers</c> and the starts are 0-based layer indices, "11,15,19,23"), <c>head.features</c>,
/// <c>head.out_channels</c> ("48,96,192,384"). Weights: the backbone's under <c>backbone.</c> (<c>patch_embed.{weight,bias}</c> as [hidden, 3·patch²],
/// <c>cls_token</c> [1, hidden], <c>camera_token</c> [1, hidden] (the reference view's), <c>position_embedding.weight</c> [1 + grid², hidden] already resampled to the grid,
/// <c>norm</c>, <c>blocks.{i}.{norm1,norm2,attn.qkv,attn.proj,attn.q_norm,attn.k_norm,ls1.gamma,ls2.gamma,mlp.fc1,mlp.fc2}</c>) and the head's
/// under <c>head.</c> with the source names (<c>norm</c>, <c>projects.{i}</c>, <c>resize_layers.{i}</c>, <c>scratch.*</c> of the main branch).
/// </remarks>
public sealed class Da3DepthAdapter : IModelAdapter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "da3_depth";

    private const float PositionRatio = 0.1f;

    /// <inheritdoc/>
    public string Name => "Depth Anything 3";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => [Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        int patch = src.Arch<int>("patch_size");
        int size = src.Arch<int>("image_size");
        var graph = BuildGraph(src, patch, size);
        var processor = JsonNode.Parse(src.Get<string>(CanonicalKeys.ImageProcessor))!;
        var description = new ModelDescription
        {
            Kind = ModelKinds.DepthEstimation,
            Components =
            [
                new ModelComponent
                {
                    Role = ModelComponentRoles.FeatureExtractor,
                    Config = new Dictionary<string, JsonElement> { [ImageProcessor.ComponentKey] = JsonSerializer.SerializeToElement(processor) },
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.VisionEncoder,
                    Graphs = [WellKnown.DepthGraph],
                    Config = ComponentConfig.Of(("patchSize", patch), ("imageSize", size)),
                },
            ],
        };
        return new ModelDefinition
        {
            Architecture = src.Architecture,
            Name = src.Get(CanonicalKeys.Name, src.Architecture),
            Weights = src.Weights,
            Graphs = new Dictionary<string, ComputeGraph> { [WellKnown.DepthGraph] = graph },
            SourceMetadata = src.Metadata,
            OwnedResources = src.OwnedResources,
            Description = description,
        };
    }

    private static int[] Ints(string text) => [.. text.Split(',').Select(int.Parse)];

    private static ComputeGraph BuildGraph(ModelSource src, int patch, int size)
    {
        int layers = src.Arch<int>("backbone.layer_count");
        int hidden = src.Arch<int>("backbone.hidden_size");
        int heads = src.Arch<int>("backbone.head_count");
        int altStart = src.Arch<int>("backbone.alt_start");
        int qkNormStart = src.Arch<int>("backbone.qknorm_start");
        int ropeStart = src.Arch<int>("backbone.rope_start");
        float ropeBase = src.Arch("backbone.rope_frequency", 100f);
        int[] outLayers = Ints(src.Arch<string>("backbone.out_layers"));
        bool catToken = src.Arch("backbone.cat_token", true);
        int[] outChannels = Ints(src.Arch<string>("head.out_channels"));
        int grid = size / patch;
        int n = grid * grid;
        int rows = n + 1;
        int headDim = hidden / heads;
        float scale = 1f / MathF.Sqrt(headDim);
        var b = new GraphBuilder(WellKnown.DepthGraph, src.Weight);
        var patches = b.Input("patches", InputRole.Media, DataType.F32, new Shape(n, 3 * patch * patch));

        var x = b.Linear(patches, "backbone.patch_embed.weight", "backbone.patch_embed.bias");
        x = b.Concat(0, b.Weight("backbone.cls_token"), x);
        x = b.Add(x, b.Weight("backbone.position_embedding.weight"));

        // 2-D rotary tables: local layers rotate by the patch's (row + 1, column + 1), global layers (one view) by the constant (1, 1); the class row is (0, 0).
        var (cosLocal, sinLocal) = RotaryTables(b, grid, headDim, ropeBase, global: false);
        var (cosGlobal, sinGlobal) = RotaryTables(b, grid, headDim, ropeBase, global: true);

        var features = new List<Value>();
        Value localX = x;
        for (int i = 0; i < layers; i++)
        {
            using var _ = b.Group($"backbone.blocks.{i}");
            if (altStart >= 0 && i == altStart)
            {
                x = b.Concat(0, b.Weight("backbone.camera_token"), b.Slice(x, 0, 1, n));
            }

            bool global = altStart >= 0 && i >= altStart && i % 2 == 1;
            bool rope = ropeStart >= 0 && i >= ropeStart;
            x = Block(b, x, $"backbone.blocks.{i}.", rows, heads, headDim, hidden, scale, qkNorm: qkNormStart >= 0 && i >= qkNormStart,
                rope ? (global ? cosGlobal : cosLocal) : null, rope ? (global ? sinGlobal : sinLocal) : null);
            if (!global)
            {
                localX = x;
            }

            if (Array.IndexOf(outLayers, i) >= 0)
            {
                var normed = LayerNorm(b, x, "backbone.norm", 1e-5f);
                var feature = catToken ? b.Concat(1, localX, normed) : normed;
                features.Add(b.Slice(feature, 0, 1, n));
            }
        }

        int c = hidden * (catToken ? 2 : 1);
        var maps = new Value[4];
        var sizes = new (int H, int W)[4];
        for (int s = 0; s < 4; s++)
        {
            using var _ = b.Group($"head.reassemble.{s}");
            var normed = LayerNorm(b, features[s], "head.norm", 1e-5f);
            var map = b.Reshape(b.Contiguous(b.Permute(normed, 1, 0)), 1, c, grid, grid);
            map = Conv(b, map, $"head.projects.{s}", 1, 0);
            map = AddPositionEmbedding(b, map, outChannels[s], grid, grid);
            switch (s)
            {
                case 0:
                    map = ConvTranspose(b, map, $"head.resize_layers.{s}", 4);
                    sizes[s] = (grid * 4, grid * 4);
                    break;
                case 1:
                    map = ConvTranspose(b, map, $"head.resize_layers.{s}", 2);
                    sizes[s] = (grid * 2, grid * 2);
                    break;
                case 2:
                    sizes[s] = (grid, grid);
                    break;
                default:
                    map = Conv(b, map, $"head.resize_layers.{s}", 2, 1);
                    sizes[s] = ((grid - 1) / 2 + 1, (grid - 1) / 2 + 1);
                    break;
            }

            maps[s] = b.Emit1("Conv", [map, b.Weight($"head.scratch.layer{s + 1}_rn.weight")], Attributes.From(("pads", new long[] { 1, 1, 1, 1 })));
        }

        Value fused;
        using (b.Group("head.fusion"))
        {
            var level = Fusion(b, "head.scratch.refinenet4", maps[3], null, sizes[2]);
            level = Fusion(b, "head.scratch.refinenet3", level, maps[2], sizes[1]);
            level = Fusion(b, "head.scratch.refinenet2", level, maps[1], sizes[0]);
            fused = Fusion(b, "head.scratch.refinenet1", level, maps[0], (sizes[0].H * 2, sizes[0].W * 2));
        }

        using (b.Group("head.output"))
        {
            int side = grid * patch;
            int half = (int)b.Weight("head.scratch.output_conv1.weight").Shape[0].Fixed;
            var y = Conv(b, fused, "head.scratch.output_conv1", 1, 1);
            y = Resize(b, y, (side, side));
            y = AddPositionEmbedding(b, y, half, side, side);
            y = b.Unary("Relu", Conv(b, y, "head.scratch.output_conv2.0", 1, 1));
            y = Conv(b, y, "head.scratch.output_conv2.2", 1, 0);
            b.Output(WellKnown.DepthOutput, b.Unary("Exp", b.Slice(y, 1, 0, 1)));
        }

        return b.Build();
    }

    private static Value Block(GraphBuilder b, Value x, string p, int rows, int heads, int headDim, int hidden, float scale, bool qkNorm, Value? cos, Value? sinSigned)
    {
        var h = LayerNorm(b, x, p + "norm1", 1e-6f);
        var qkv = b.Linear(h, p + "attn.qkv.weight", p + "attn.qkv.bias");
        var q = b.Reshape(b.Contiguous(b.Slice(qkv, 1, 0, hidden)), rows, heads, headDim);
        var k = b.Reshape(b.Contiguous(b.Slice(qkv, 1, hidden, hidden)), rows, heads, headDim);
        var v = b.Reshape(b.Contiguous(b.Slice(qkv, 1, 2L * hidden, hidden)), rows, heads, headDim);
        if (qkNorm)
        {
            q = LayerNorm(b, q, p + "attn.q_norm", 1e-5f);
            k = LayerNorm(b, k, p + "attn.k_norm", 1e-5f);
        }

        if (cos is not null)
        {
            q = Rotate(b, q, cos, sinSigned!, headDim);
            k = Rotate(b, k, cos, sinSigned!, headDim);
        }

        var attn = b.Emit1("Attention", [q, k, v], Attributes.From(("scale", scale), ("causal", false)), $"{p}attn");
        var o = b.Linear(b.Reshape(attn, rows, hidden), p + "attn.proj.weight", p + "attn.proj.bias");
        x = b.Add(x, b.Mul(o, b.Weight(p + "ls1.gamma")), $"{p}attn_residual");
        h = LayerNorm(b, x, p + "norm2", 1e-6f);
        var mlp = b.Linear(b.Unary("Gelu", b.Linear(h, p + "mlp.fc1.weight", p + "mlp.fc1.bias")), p + "mlp.fc2.weight", p + "mlp.fc2.bias");
        return b.Add(x, b.Mul(mlp, b.Weight(p + "ls2.gamma")), $"{p}l_out");
    }

    private static Value LayerNorm(GraphBuilder b, Value x, string name, float eps)
        => b.Emit1("LayerNorm", [x, b.Weight(name + ".weight"), b.Weight(name + ".bias")], Attributes.From(("eps", eps)));

    // [rows, 1, D] tables over the head: the first half of the features rotates with the row, the second with the column; each half is
    // [x1 | x2] → [-x2 | x1] (rotate_half), so that rotated = x · cos + swap(x) · sinSigned.
    private static (Value Cos, Value SinSigned) RotaryTables(GraphBuilder b, int grid, int headDim, float baseFrequency, bool global)
    {
        int rows = (grid * grid) + 1;
        int feature = headDim / 2;
        int quarter = feature / 2;
        var cos = new float[rows * headDim];
        var sin = new float[rows * headDim];
        for (int r = 0; r < rows; r++)
        {
            int py = r == 0 ? 0 : global ? 1 : ((r - 1) / grid) + 1;
            int px = r == 0 ? 0 : global ? 1 : ((r - 1) % grid) + 1;
            for (int axis = 0; axis < 2; axis++)
            {
                int pos = axis == 0 ? py : px;
                for (int j = 0; j < quarter; j++)
                {
                    float inv = (float)(1.0 / Math.Pow(baseFrequency, (double)(2 * j) / feature));
                    float angle = pos * inv;
                    float cs = MathF.Cos(angle);
                    float sn = MathF.Sin(angle);
                    int d = (r * headDim) + (axis * feature) + j;
                    cos[d] = cs;
                    cos[d + quarter] = cs;
                    sin[d] = -sn;
                    sin[d + quarter] = sn;
                }
            }
        }

        var shape = new Shape(rows, 1, headDim);
        return (b.Constant("rope_cos", DataType.F32, shape, Bytes(cos)), b.Constant("rope_sin", DataType.F32, shape, Bytes(sin)));
    }

    private static Value Rotate(GraphBuilder b, Value x, Value cos, Value sinSigned, int headDim)
    {
        int f = headDim / 2;
        int q = f / 2;
        var swapped = b.Concat(2, b.Slice(x, 2, q, q), b.Slice(x, 2, 0, q), b.Slice(x, 2, f + q, q), b.Slice(x, 2, f, q));
        return b.Add(b.Mul(x, cos), b.Mul(swapped, sinSigned));
    }

    // The DPT feature fusion block: lateral residual unit added to the top branch, a residual unit, bilinear resize (corners aligned) and a 1x1 convolution.
    private static Value Fusion(GraphBuilder b, string p, Value top, Value? lateral, (int H, int W) size)
    {
        var y = top;
        if (lateral is not null)
        {
            y = b.Add(y, ResidualUnit(b, lateral, p + ".resConfUnit1."));
        }

        y = ResidualUnit(b, y, p + ".resConfUnit2.");
        return Conv(b, Resize(b, y, size), p + ".out_conv", 1, 0);
    }

    private static Value ResidualUnit(GraphBuilder b, Value x, string p)
    {
        var y = Conv(b, b.Unary("Relu", x), p + "conv1", 1, 1);
        y = Conv(b, b.Unary("Relu", y), p + "conv2", 1, 1);
        return b.Add(y, x);
    }

    // The UV grid embedding of the head: channels [sin(x·w) | cos(x·w) | sin(y·w) | cos(y·w)] with w = 100^(-j / (C / 4)), scaled by 0.1, on a grid whose
    // corners sit at ±1/√2 (a square image). Added to the first half of the channels along the width and to the second half along the height.
    private static Value AddPositionEmbedding(GraphBuilder b, Value x, int channels, int height, int width)
    {
        int quarter = channels / 4;
        int half = channels / 2;
        float span = 1f / MathF.Sqrt(2f);
        float[] Coordinates(int count)
        {
            var values = new float[count];
            float end = span * (count - 1) / count;
            for (int i = 0; i < count; i++)
            {
                values[i] = count == 1 ? 0f : -end + (2f * end * i / (count - 1));
            }

            return values;
        }

        var cx = Coordinates(width);
        var cy = Coordinates(height);
        var px = new float[half * width];
        var py = new float[half * height];
        for (int j = 0; j < quarter; j++)
        {
            double omega = 1.0 / Math.Pow(100.0, (double)j / quarter);
            for (int i = 0; i < width; i++)
            {
                double a = cx[i] * omega;
                px[(j * width) + i] = (float)(Math.Sin(a) * PositionRatio);
                px[((quarter + j) * width) + i] = (float)(Math.Cos(a) * PositionRatio);
            }

            for (int i = 0; i < height; i++)
            {
                double a = cy[i] * omega;
                py[(j * height) + i] = (float)(Math.Sin(a) * PositionRatio);
                py[((quarter + j) * height) + i] = (float)(Math.Cos(a) * PositionRatio);
            }
        }

        var xs = b.Constant("pos_x", DataType.F32, new Shape(1, half, 1, width), Bytes(px));
        var ys = b.Constant("pos_y", DataType.F32, new Shape(1, half, height, 1), Bytes(py));
        var first = b.Add(b.Contiguous(b.Slice(x, 1, 0, half)), xs);
        var second = b.Add(b.Contiguous(b.Slice(x, 1, half, half)), ys);
        return b.Concat(1, first, second);
    }

    private static Value Conv(GraphBuilder b, Value x, string name, int stride, int pad)
        => b.Emit1("Conv", [x, b.Weight(name + ".weight"), b.Weight(name + ".bias")],
            Attributes.From(("strides", new long[] { stride, stride }), ("pads", new long[] { pad, pad, pad, pad })));

    private static Value ConvTranspose(GraphBuilder b, Value x, string name, int stride)
        => b.Emit1("ConvTranspose", [x, b.Weight(name + ".weight"), b.Weight(name + ".bias")],
            Attributes.From(("strides", new long[] { stride, stride }), ("pads", new long[] { 0, 0, 0, 0 })));

    private static Value Resize(GraphBuilder b, Value x, (int H, int W) size)
        => b.Emit1("Resize", [x], Attributes.From(("size", new long[] { size.H, size.W }), ("mode", "linear"), ("coord", "align_corners")));

    private static byte[] Bytes(float[] values)
    {
        var bytes = new byte[values.Length * 4];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes;
    }
}
