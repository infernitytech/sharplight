using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Sharplight.Graph;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// RT-DETR real-time object detectors (canonical arch <c>rt_detr</c>): a ResNet-D backbone, a hybrid encoder (AIFI transformer on the last level, a
/// top-down feature pyramid and a bottom-up path aggregation of CSP-RepVGG blocks), top-k query selection over all feature positions with anchors, and a
/// decoder of multi-scale deformable attention (implemented with <c>GridSample</c>) with iterative box refinement. Builds
/// <see cref="WellKnown.DetectGraph"/> for a fixed square input; sigmoid class scores.
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>rt_detr.*</c>): <c>hidden_size</c>, <c>attention.head_count</c>, <c>aifi.layer_count</c>, <c>aifi.ffn_hidden_size</c>,
/// <c>aifi.mlp.activation</c>, <c>decoder.layer_count</c>, <c>decoder.ffn_hidden_size</c>, <c>decoder.mlp.activation</c>, <c>decoder.point_count</c>,
/// <c>mlp.activation</c> (convolution blocks), <c>query_count</c>, <c>image_size</c>, <c>level_count</c> (3), <c>strides</c> ("8,16,32"),
/// <c>backbone.depths</c>, <c>hybrid.block_count</c> (3), <c>norm_eps</c>. Weights are listed on <c>RtDetrHfMapping</c> (the names it produces);
/// every convolution carries a folded batch norm.
/// </remarks>
public sealed class RtDetrAdapter : IModelAdapter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "rt_detr";

    /// <inheritdoc/>
    public string Name => "RT-DETR object detector";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => [Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        var graph = new Builder(src).Build();
        var labels = src.Get(CanonicalKeys.ClassifierLabels, Array.Empty<string>());
        var processor = JsonNode.Parse(src.Get<string>(CanonicalKeys.ImageProcessor))!;
        var description = new ModelDescription
        {
            Kind = ModelKinds.ObjectDetection,
            Config = ComponentConfig.Of(("scoring", "sigmoid")).Concat(new Dictionary<string, JsonElement> { ["labels"] = JsonSerializer.SerializeToElement(labels) }).ToDictionary(),
            Components =
            [
                new ModelComponent
                {
                    Role = ModelComponentRoles.FeatureExtractor,
                    Config = new Dictionary<string, JsonElement> { [ImageProcessor.ComponentKey] = JsonSerializer.SerializeToElement(processor) },
                },
                new ModelComponent { Role = ModelComponentRoles.VisionEncoder, Graphs = [WellKnown.DetectGraph], Config = ComponentConfig.Of(("imageSize", src.Arch<int>("image_size"))) },
            ],
        };
        return new ModelDefinition
        {
            Architecture = src.Architecture,
            Name = src.Get(CanonicalKeys.Name, src.Architecture),
            Weights = src.Weights,
            Graphs = new Dictionary<string, ComputeGraph> { [WellKnown.DetectGraph] = graph },
            SourceMetadata = src.Metadata,
            OwnedResources = src.OwnedResources,
            Description = description,
        };
    }

    private sealed class Builder
    {
        private readonly ModelSource _src;
        private readonly GraphBuilder _b;
        private readonly int _d;
        private readonly int _heads;
        private readonly int _size;
        private readonly int _levels;
        private readonly int _points;
        private readonly int[] _strides;
        private readonly string _convAct;
        private readonly int _blocks;
        private readonly float _eps;

        public Builder(ModelSource src)
        {
            _src = src;
            _b = new GraphBuilder(WellKnown.DetectGraph, src.Weight);
            _d = src.Arch<int>("hidden_size");
            _heads = src.Arch<int>("attention.head_count");
            _size = src.Arch<int>("image_size");
            _levels = src.Arch("level_count", 3);
            _points = src.Arch<int>("decoder.point_count");
            _strides = [.. src.Arch("strides", "8,16,32").Split(',').Select(int.Parse)];
            _convAct = src.Arch("mlp.activation", "silu");
            _blocks = src.Arch("hybrid.block_count", 3);
            _eps = src.Arch("norm_eps", 1e-5f);
        }

        public ComputeGraph Build()
        {
            var b = _b;
            var pixels = b.Input("pixels", InputRole.Media, DataType.F32, new Shape(1, 3, _size, _size));
            var depths = _src.Arch<string>("backbone.depths").Split(',').Select(int.Parse).ToArray();
            var stages = ResNetBackbone.Build(b, pixels, new ResNetOptions("backbone.", depths, _src.Arch("backbone.bottleneck", false), true, true, _src.Arch("backbone.stride_in_first_conv", false)));
            var maps = new Value[_levels];
            var shapes = new (int H, int W)[_levels];
            for (int i = 0; i < _levels; i++)
            {
                shapes[i] = (_size / _strides[i], _size / _strides[i]);
                maps[i] = ResNetBackbone.Conv(b, stages[stages.Length - _levels + i], $"encoder_proj.{i}", 1, 0, relu: false);
            }

            int aifi = _src.Arch("aifi.layer_count", 1);
            if (aifi > 0)
            {
                maps[_levels - 1] = Aifi(maps[_levels - 1], shapes[_levels - 1]);
            }

            var pan = HybridEncoder(maps, shapes);
            var sources = new Value[_levels];
            for (int i = 0; i < _levels; i++)
            {
                var projected = ResNetBackbone.Conv(b, pan[i], $"decoder_proj.{i}", 1, 0, relu: false);
                sources[i] = b.Contiguous(b.Permute(b.Reshape(projected, _d, shapes[i].H * shapes[i].W), 1, 0));
            }

            var source = b.Concat(0, sources);
            int total = shapes.Sum(s => s.H * s.W);
            using (b.Group("decoder"))
            {
                Decode(source, shapes, total);
            }

            return b.Build();
        }

        private Value Conv(Value x, string name, int stride, int pad, bool act)
        {
            var y = ResNetBackbone.Conv(_b, x, name, stride, pad, relu: false);
            return act ? TowerBlocks.Activation(_b, y, _convAct) : y;
        }

        private Value CspRep(Value x, string p)
        {
            var h1 = Conv(x, p + "conv1", 1, 0, true);
            for (int j = 0; j < _blocks; j++)
            {
                string r = $"{p}rep.{j}.";
                h1 = TowerBlocks.Activation(_b, _b.Add(Conv(h1, r + "conv1", 1, 1, false), Conv(h1, r + "conv2", 1, 0, false)), _convAct);
            }

            var h2 = Conv(x, p + "conv2", 1, 0, true);
            var sum = _b.Add(h1, h2);
            return _b.HasWeight(p + "conv3.weight") ? Conv(sum, p + "conv3", 1, 0, true) : sum;
        }

        private Value[] HybridEncoder(Value[] maps, (int H, int W)[] shapes)
        {
            var b = _b;
            int stages = _levels - 1;
            var fpn = new List<Value> { maps[^1] };
            for (int i = 0; i < stages; i++)
            {
                var backbone = maps[stages - i - 1];
                var top = Conv(fpn[^1], $"hybrid.lateral.{i}", 1, 0, true);
                fpn[^1] = top;
                var (h, w) = shapes[stages - i - 1];
                var up = b.Emit1("Resize", [top], Attributes.From(("size", new long[] { h, w }), ("mode", "nearest"), ("coord", "asymmetric"), ("nearest_mode", "floor")));
                fpn.Add(CspRep(b.Concat(1, up, backbone), $"hybrid.fpn.{i}."));
            }

            fpn.Reverse();
            var pan = new List<Value> { fpn[0] };
            for (int i = 0; i < stages; i++)
            {
                var down = Conv(pan[^1], $"hybrid.down.{i}", 2, 1, true);
                pan.Add(CspRep(b.Concat(1, down, fpn[i + 1]), $"hybrid.pan.{i}."));
            }

            return [.. pan];
        }

        private Value Aifi(Value map, (int H, int W) shape)
        {
            var b = _b;
            int n = shape.H * shape.W;
            var x = b.Contiguous(b.Permute(b.Reshape(map, _d, n), 1, 0));
            var pos = b.Constant("aifi_position", DataType.F32, new Shape(n, _d), Bytes(Position(shape.H, shape.W)));
            int layers = _src.Arch("aifi.layer_count", 1);
            string act = _src.Arch("aifi.mlp.activation", "gelu");
            for (int l = 0; l < layers; l++)
            {
                string p = $"aifi.{l}.";
                var qk = b.Add(x, pos);
                x = LayerNorm(b.Add(x, Attention(qk, qk, x, p + "self_attn.", n, n)), p + "self_attn_norm");
                var ffn = b.Linear(TowerBlocks.Activation(b, b.Linear(x, p + "ffn.up.weight", p + "ffn.up.bias"), act), p + "ffn.down.weight", p + "ffn.down.bias");
                x = LayerNorm(b.Add(x, ffn), p + "ffn_norm");
            }

            return b.Reshape(b.Contiguous(b.Permute(x, 1, 0)), 1, _d, shape.H, shape.W);
        }

        private void Decode(Value source, (int H, int W)[] shapes, int total)
        {
            var b = _b;
            var (anchors, valid) = Anchors(shapes);
            var memory = b.Mul(source, b.Constant("valid_mask", DataType.F32, new Shape(total, 1), Bytes(valid)));
            var outputMemory = LayerNorm(b.Linear(memory, "enc_output.linear.weight", "enc_output.linear.bias"), "enc_output.norm");
            var scores = b.Linear(outputMemory, "enc_score_head.weight", "enc_score_head.bias");
            var coordinates = b.Add(Mlp(outputMemory, "enc_box_head.", 3), b.Constant("anchors", DataType.F32, new Shape(total, 4), Bytes(anchors)));
            int queries = _src.Arch<int>("query_count");
            var best = b.Reshape(b.Emit1("ReduceMax", [scores]), total);
            var top = b.Emit("TopK", [best], Attributes.From(("k", (long)queries)));
            var indices = top[1];
            // The device gather moves rows of a multiple of 8 values: the 4 box values travel in an 8-wide row.
            var padded = b.Concat(1, coordinates, b.Constant("gather_padding", DataType.F32, new Shape(total, 4), new byte[total * 16]));
            var reference = b.Sigmoid(b.Slice(b.Gather(padded, indices), 1, 0, 4));
            var h = b.Gather(outputMemory, indices);
            int layers = _src.Arch<int>("decoder.layer_count");
            string act = _src.Arch("decoder.mlp.activation", "relu");
            Value logits = h;
            for (int il = 0; il < layers; il++)
            {
                using var _ = b.Group($"layers.{il}");
                string p = $"decoder.layers.{il}.";
                var qpos = Mlp(reference, "decoder.query_pos.", 2);
                var qk = b.Add(h, qpos);
                h = LayerNorm(b.Add(h, Attention(qk, qk, h, p + "self_attn.", queries, queries)), p + "self_attn_norm");
                h = LayerNorm(b.Add(h, Deformable(b.Add(h, qpos), source, reference, p + "cross_attn.", shapes, queries)), p + "cross_attn_norm");
                var ffn = b.Linear(TowerBlocks.Activation(b, b.Linear(h, p + "ffn.up.weight", p + "ffn.up.bias"), act), p + "ffn.down.weight", p + "ffn.down.bias");
                h = LayerNorm(b.Add(h, ffn), p + "ffn_norm");
                var delta = Mlp(h, $"decoder.box_head.{il}.", 3);
                reference = b.Sigmoid(b.Add(delta, InverseSigmoid(reference)));
                if (il == layers - 1)
                {
                    logits = b.Linear(h, $"decoder.class_head.{il}.weight", $"decoder.class_head.{il}.bias");
                }
            }

            b.Output(WellKnown.LogitsOutput, logits);
            b.Output(WellKnown.BoxesOutput, reference);
        }

        private Value Deformable(Value query, Value source, Value reference, string p, (int H, int W)[] shapes, int queries)
        {
            var b = _b;
            int hd = _d / _heads;
            var value = b.Linear(source, p + "value_proj.weight", p + "value_proj.bias");
            var offsets = b.Reshape(b.Linear(query, p + "sampling_offsets.weight", p + "sampling_offsets.bias"), queries, _heads, _levels, _points, 2);
            var weights = b.Emit1("Softmax", [b.Reshape(b.Linear(query, p + "attention_weights.weight", p + "attention_weights.bias"), queries, _heads, _levels * _points)]);
            weights = b.Reshape(b.Contiguous(b.Permute(weights, 1, 0, 2)), _heads, 1, queries, _levels * _points);
            var center = b.Reshape(b.Slice(reference, 1, 0, 2), queries, 1, 1, 1, 2);
            var size = b.Reshape(b.Slice(reference, 1, 2, 2), queries, 1, 1, 1, 2);
            var location = b.Add(center, b.Mul(b.Scale(offsets, 0.5f / _points), size));
            var grid = b.Scale(location, 2f, -1f);
            var sampled = new Value[_levels];
            int start = 0;
            for (int l = 0; l < _levels; l++)
            {
                var (h, w) = shapes[l];
                var level = b.Slice(value, 0, start, h * w);
                level = b.Reshape(b.Contiguous(b.Permute(b.Reshape(level, h * w, _heads, hd), 1, 2, 0)), _heads, hd, h, w);
                var g = b.Reshape(b.Slice(grid, 2, l, 1), queries, _heads, _points, 2);
                g = b.Contiguous(b.Permute(g, 1, 0, 2, 3));
                sampled[l] = b.Emit1("GridSample", [level, g], Attributes.From(("mode", "linear"), ("padding", "zeros"), ("align_corners", false)));
                start += h * w;
            }

            var all = b.Concat(3, sampled);
            var weighted = b.Emit1("Reduce", [b.Mul(all, weights)], Attributes.From(("mode", "sum"), ("axes", new long[] { 3 }), ("keepdims", false)));
            var merged = b.Reshape(b.Contiguous(b.Permute(weighted, 2, 0, 1)), queries, _d);
            return b.Linear(merged, p + "output_proj.weight", p + "output_proj.bias");
        }

        private Value InverseSigmoid(Value x)
        {
            var b = _b;
            var x1 = b.Emit1("Clip", [x], Attributes.From(("min", 1e-5f)));
            var x2 = b.Emit1("Clip", [b.Scale(x, -1f, 1f)], Attributes.From(("min", 1e-5f)));
            return b.Unary("Log", b.Binary("Div", x1, x2));
        }

        private Value Attention(Value query, Value key, Value value, string p, int qRows, int kRows)
        {
            var b = _b;
            int hd = _d / _heads;
            var q = b.Reshape(b.Linear(query, p + "q.weight", p + "q.bias"), qRows, _heads, hd);
            var k = b.Reshape(b.Linear(key, p + "k.weight", p + "k.bias"), kRows, _heads, hd);
            var v = b.Reshape(b.Linear(value, p + "v.weight", p + "v.bias"), kRows, _heads, hd);
            var a = b.Emit1("Attention", [q, k, v], Attributes.From(("scale", 1f / MathF.Sqrt(hd)), ("causal", false)));
            return b.Linear(b.Reshape(a, qRows, _d), p + "o.weight", p + "o.bias");
        }

        private Value Mlp(Value x, string p, int layers)
        {
            for (int i = 0; i < layers; i++)
            {
                x = _b.Linear(x, $"{p}{i}.weight", $"{p}{i}.bias");
                if (i < layers - 1)
                {
                    x = _b.Unary("Relu", x);
                }
            }

            return x;
        }

        private Value LayerNorm(Value x, string name) => TowerBlocks.LayerNorm(_b, x, name, _eps);

        // [sin(h) | cos(h) | sin(w) | cos(w)] with geometric frequencies, row-major over the grid.
        private float[] Position(int height, int width)
        {
            int dim = _d / 4;
            var result = new float[height * width * _d];
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int row = (y * width + x) * _d;
                    for (int i = 0; i < dim; i++)
                    {
                        double omega = 1.0 / Math.Pow(10000.0, (double)i / dim);
                        result[row + i] = (float)Math.Sin(y * omega);
                        result[row + dim + i] = (float)Math.Cos(y * omega);
                        result[row + 2 * dim + i] = (float)Math.Sin(x * omega);
                        result[row + 3 * dim + i] = (float)Math.Cos(x * omega);
                    }
                }
            }

            return result;
        }

        // Anchor boxes (as logits) of every feature position, and the mask of anchors inside (0.01, 0.99).
        private static (float[] Anchors, float[] Valid) Anchors((int H, int W)[] shapes)
        {
            var anchors = new List<float>();
            var valid = new List<float>();
            for (int level = 0; level < shapes.Length; level++)
            {
                var (h, w) = shapes[level];
                for (int y = 0; y < h; y++)
                {
                    for (int x = 0; x < w; x++)
                    {
                        float[] a = [(x + 0.5f) / w, (y + 0.5f) / h, 0.05f * MathF.Pow(2, level), 0.05f * MathF.Pow(2, level)];
                        bool ok = a.All(v => v > 0.01f && v < 0.99f);
                        valid.Add(ok ? 1f : 0f);
                        anchors.AddRange(a.Select(v => ok ? MathF.Log(v / (1 - v)) : float.MaxValue));
                    }
                }
            }

            return ([.. anchors], [.. valid]);
        }

        private static byte[] Bytes(float[] values) => MemoryMarshal.AsBytes(values.AsSpan()).ToArray();
    }
}
