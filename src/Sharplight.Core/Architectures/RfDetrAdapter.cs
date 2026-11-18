using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Sharplight.Graph;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// RF-DETR object detectors (canonical arch <c>rf_detr</c>): a windowed DINOv2 backbone (the patch grid is split into <c>windows²</c> windows with a class token each;
/// the layers in <c>backbone.window_blocks</c> attend inside a window, the others across all of them), a C2f projector over the concatenated feature maps of four layers,
/// two-stage query selection (top-k of the encoder's class scores, boxes refined from anchors) and a decoder of self-attention and single-level
/// deformable cross-attention (implemented with <c>GridSample</c>) whose boxes are refined once from the selected reference boxes. Builds <see cref="WellKnown.DetectGraph"/>
/// for a fixed square input; sigmoid class scores; only the first query group (inference) is read.
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>rf_detr.*</c>): <c>image_size</c>, <c>patch_size</c>, <c>hidden_size</c>, <c>query_count</c>, <c>backbone.{layer_count,hidden_size,head_count,window_count,
/// window_blocks,out_layers,norm_eps}</c> (<c>out_layers</c>: 1-based layer counts "3,6,9,12"), <c>projector.{block_count,expansion}</c>, <c>decoder.{layer_count,self_head_count,
/// cross_head_count,point_count,ffn_hidden_size}</c>, <c>norm_eps</c>. Weights: <c>backbone.{patch_embed.{weight,bias},cls_token,position_embedding.weight,norm,layers.i.{norm1,norm2,
/// attn.{q,k,v,o},ls1.weight,ls2.weight,mlp.{up,down}}}</c>, <c>projector.{conv1,conv2,bottlenecks.j.conv{1,2}}.{weight,norm.weight,norm.bias}</c>, <c>projector.norm</c>,
/// <c>enc_output.{weight,bias}</c>, <c>enc_output_norm</c>, <c>enc_class_head</c>, <c>enc_box_head.j</c>, <c>query_feat</c> and <c>reference_embed</c> ([queries, ..]), and the
/// decoder's as for <c>rt_detr</c> (<c>decoder.layers.i.*</c>, <c>decoder.norm</c>, <c>decoder.query_pos.j</c>, <c>class_head</c>, <c>box_head.j</c>).
/// </remarks>
public sealed class RfDetrAdapter : IModelAdapter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "rf_detr";

    /// <inheritdoc/>
    public string Name => "RF-DETR object detector";

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
        private readonly int _size;
        private readonly int _patch;
        private readonly int _grid;
        private readonly int _d;
        private readonly int _queries;
        private readonly int _bbHidden;
        private readonly int _bbHeads;
        private readonly int _bbLayers;
        private readonly int _windows;
        private readonly int[] _windowBlocks;
        private readonly int[] _outLayers;
        private readonly float _bbEps;
        private readonly float _eps;

        public Builder(ModelSource src)
        {
            _src = src;
            _b = new GraphBuilder(WellKnown.DetectGraph, src.Weight);
            _size = src.Arch<int>("image_size");
            _patch = src.Arch<int>("patch_size");
            _grid = _size / _patch;
            _d = src.Arch<int>("hidden_size");
            _queries = src.Arch<int>("query_count");
            _bbHidden = src.Arch<int>("backbone.hidden_size");
            _bbHeads = src.Arch<int>("backbone.head_count");
            _bbLayers = src.Arch<int>("backbone.layer_count");
            _windows = src.Arch<int>("backbone.window_count");
            _windowBlocks = [.. src.Arch("backbone.window_blocks", "").Split(',', StringSplitOptions.RemoveEmptyEntries).Select(int.Parse)];
            _outLayers = [.. src.Arch<string>("backbone.out_layers").Split(',').Select(int.Parse)];
            _bbEps = src.Arch("backbone.norm_eps", 1e-6f);
            _eps = src.Arch("norm_eps", 1e-5f);
        }

        public ComputeGraph Build()
        {
            var b = _b;
            var pixels = b.Input("pixels", InputRole.Media, DataType.F32, new Shape(1, 3, _size, _size));
            var maps = Backbone(pixels);
            var feature = Projector(b.Concat(1, maps));
            int n = _grid * _grid;
            var source = b.Contiguous(b.Permute(b.Reshape(feature, _d, n), 1, 0));
            using (b.Group("decoder"))
            {
                Decode(source, n);
            }

            return b.Build();
        }

        // ---- windowed DINOv2 ----

        private Value[] Backbone(Value pixels)
        {
            var b = _b;
            using var _ = b.Group("backbone");
            int n = _grid * _grid;
            int per = _grid / _windows;
            int rowsPerWindow = (per * per) + 1;
            int rows = _windows * _windows * rowsPerWindow;
            var conv = b.Emit1("Conv", [pixels, b.Weight("backbone.patch_embed.weight"), b.Weight("backbone.patch_embed.bias")],
                Attributes.From(("strides", new long[] { _patch, _patch }), ("pads", new long[] { 0, 0, 0, 0 })));
            var patches = b.Contiguous(b.Permute(b.Reshape(conv, _bbHidden, n), 1, 0));
            var position = b.Weight("backbone.position_embedding.weight");
            patches = b.Add(patches, b.Slice(position, 0, 1, n));
            var cls = b.Add(b.Slice(position, 0, 0, 1), b.Weight("backbone.cls_token"));
            var windowed = b.Reshape(b.Contiguous(b.Permute(b.Reshape(patches, _windows, per, _windows, per, _bbHidden), 0, 2, 1, 3, 4)), _windows * _windows, per * per, _bbHidden);
            var clsRows = b.Reshape(b.Concat(0, Enumerable.Repeat(cls, _windows * _windows).ToArray()), _windows * _windows, 1, _bbHidden);
            var x = b.Reshape(b.Concat(1, clsRows, windowed), rows, _bbHidden);

            int hd = _bbHidden / _bbHeads;
            var maps = new List<Value>();
            for (int i = 0; i < _bbLayers; i++)
            {
                using var g = b.Group($"layers.{i}");
                string p = $"backbone.layers.{i}.";
                var h = TowerBlocks.LayerNorm(b, x, p + "norm1", _bbEps);
                var q = b.Reshape(b.Linear(h, p + "attn.q.weight", p + "attn.q.bias"), rows, _bbHeads, hd);
                var k = b.Reshape(b.Linear(h, p + "attn.k.weight", p + "attn.k.bias"), rows, _bbHeads, hd);
                var v = b.Reshape(b.Linear(h, p + "attn.v.weight", p + "attn.v.bias"), rows, _bbHeads, hd);
                var attributes = Array.IndexOf(_windowBlocks, i) >= 0
                    ? Attributes.From(("scale", 1f / MathF.Sqrt(hd)), ("causal", false), ("sequence_rows", (long)rowsPerWindow), ("sequence_keys", (long)rowsPerWindow))
                    : Attributes.From(("scale", 1f / MathF.Sqrt(hd)), ("causal", false));
                var attn = b.Emit1("Attention", [q, k, v], attributes, $"{p}attn");
                var o = b.Linear(b.Reshape(attn, rows, _bbHidden), p + "attn.o.weight", p + "attn.o.bias");
                x = b.Add(x, b.Mul(o, b.Weight(p + "ls1.weight")), $"{p}attn_residual");
                h = TowerBlocks.LayerNorm(b, x, p + "norm2", _bbEps);
                var mlp = b.Linear(b.Unary("Gelu", b.Linear(h, p + "mlp.up.weight", p + "mlp.up.bias")), p + "mlp.down.weight", p + "mlp.down.bias");
                x = b.Add(x, b.Mul(mlp, b.Weight(p + "ls2.weight")), $"{p}l_out");
                if (Array.IndexOf(_outLayers, i + 1) >= 0)
                {
                    var normed = TowerBlocks.LayerNorm(b, x, "backbone.norm", _bbEps);
                    var inWindows = b.Slice(b.Reshape(normed, _windows * _windows, rowsPerWindow, _bbHidden), 1, 1, per * per);
                    var grid = b.Reshape(b.Contiguous(b.Permute(b.Reshape(b.Contiguous(inWindows), _windows, _windows, per, per, _bbHidden), 0, 2, 1, 3, 4)), n, _bbHidden);
                    maps.Add(b.Reshape(b.Contiguous(b.Permute(grid, 1, 0)), 1, _bbHidden, _grid, _grid));
                }
            }

            return [.. maps];
        }

        // ---- C2f projector ----

        private Value Projector(Value x)
        {
            var b = _b;
            using var _ = b.Group("projector");
            int blocks = _src.Arch("projector.block_count", 3);
            int hidden = (int)(_d * _src.Arch("projector.expansion", 0.5f));
            var y = ConvNorm(x, "projector.conv1", 1);
            var parts = new List<Value> { b.Contiguous(b.Slice(y, 1, 0, hidden)), b.Contiguous(b.Slice(y, 1, hidden, hidden)) };
            var h = parts[1];
            for (int j = 0; j < blocks; j++)
            {
                h = ConvNorm(ConvNorm(h, $"projector.bottlenecks.{j}.conv1", 3), $"projector.bottlenecks.{j}.conv2", 3);
                parts.Add(h);
            }

            y = ConvNorm(b.Concat(1, [.. parts]), "projector.conv2", 1);
            return ChannelNorm(y, "projector.norm", 1e-6f);
        }

        private Value ConvNorm(Value x, string name, int kernel)
        {
            var b = _b;
            int pad = kernel / 2;
            var y = b.Emit1("Conv", [x, b.Weight(name + ".weight")], Attributes.From(("pads", new long[] { pad, pad, pad, pad })));
            return b.Silu(ChannelNorm(y, name + ".norm", _eps));
        }

        // LayerNorm over the channels of [1, C, H, W].
        private Value ChannelNorm(Value x, string name, float eps)
        {
            var b = _b;
            long c = x.Shape[1].Fixed;
            long h = x.Shape[2].Fixed;
            long w = x.Shape[3].Fixed;
            var rows = b.Contiguous(b.Permute(b.Reshape(x, c, h * w), 1, 0));
            var normed = TowerBlocks.LayerNorm(b, rows, name, eps);
            return b.Reshape(b.Contiguous(b.Permute(normed, 1, 0)), 1, c, h, w);
        }

        // ---- two-stage selection and the decoder ----

        private void Decode(Value source, int total)
        {
            var b = _b;
            var (proposals, _) = Proposals();
            var propXy = b.Constant("proposal_xy", DataType.F32, new Shape(total, 2), Bytes(proposals.SelectMany(p => new[] { p[0], p[1] }).ToArray()));
            var propWh = b.Constant("proposal_wh", DataType.F32, new Shape(total, 2), Bytes(proposals.SelectMany(p => new[] { p[2], p[3] }).ToArray()));
            var memory = TowerBlocks.LayerNorm(b, b.Linear(source, "enc_output.weight", "enc_output.bias"), "enc_output_norm", _eps);
            var scores = b.Linear(memory, "enc_class_head.weight", "enc_class_head.bias");
            var delta = Mlp(memory, "enc_box_head.", 3);
            var coordinates = Refine(propXy, propWh, delta);
            var best = b.Reshape(b.Emit1("ReduceMax", [scores]), total);
            var top = b.Emit("TopK", [best], Attributes.From(("k", (long)_queries)));
            var indices = top[1];
            // The device gather moves rows of a multiple of 8 values: the 4 box values travel in an 8-wide row.
            var padded = b.Concat(1, coordinates, b.Constant("gather_padding", DataType.F32, new Shape(total, 4), new byte[total * 16]));
            var selected = b.Contiguous(b.Slice(b.Gather(padded, indices), 1, 0, 4));
            var embed = b.Weight("reference_embed");
            var reference = Refine(b.Contiguous(b.Slice(selected, 1, 0, 2)), b.Contiguous(b.Slice(selected, 1, 2, 2)), embed);
            var h = b.Weight("query_feat");
            var qpos = Mlp(SineEmbedding(reference), "decoder.query_pos.", 2);
            int layers = _src.Arch<int>("decoder.layer_count");
            int selfHeads = _src.Arch<int>("decoder.self_head_count");
            for (int il = 0; il < layers; il++)
            {
                using var _ = b.Group($"layers.{il}");
                string p = $"decoder.layers.{il}.";
                var qk = b.Add(h, qpos);
                h = LayerNorm(b.Add(h, Attention(qk, qk, h, p + "self_attn.", selfHeads)), p + "self_attn_norm");
                h = LayerNorm(b.Add(h, Deformable(b.Add(h, qpos), source, reference, p + "cross_attn.")), p + "cross_attn_norm");
                var ffn = b.Linear(b.Unary("Relu", b.Linear(h, p + "ffn.up.weight", p + "ffn.up.bias")), p + "ffn.down.weight", p + "ffn.down.bias");
                h = LayerNorm(b.Add(h, ffn), p + "ffn_norm");
            }

            var normed = LayerNorm(h, "decoder.norm");
            b.Output(WellKnown.LogitsOutput, b.Linear(normed, "decoder.class_head.weight", "decoder.class_head.bias"));
            var boxDelta = Mlp(normed, "decoder.box_head.", 3);
            b.Output(WellKnown.BoxesOutput, Refine(b.Contiguous(b.Slice(reference, 1, 0, 2)), b.Contiguous(b.Slice(reference, 1, 2, 2)), boxDelta));
        }

        // refine_bboxes: centers move by delta · size, sizes scale by exp(delta).
        private Value Refine(Value centerOrXy, Value size, Value delta)
        {
            var b = _b;
            var xy = b.Add(b.Mul(b.Contiguous(b.Slice(delta, 1, 0, 2)), size), centerOrXy);
            var wh = b.Mul(b.Unary("Exp", b.Contiguous(b.Slice(delta, 1, 2, 2))), size);
            return b.Concat(1, xy, wh);
        }

        // Sinusoidal embedding of the reference boxes ordered (y, x, w, h): per coordinate [sin, cos] interleaved over d/2 channels (temperature 10000).
        private Value SineEmbedding(Value reference)
        {
            var b = _b;
            int features = _d / 2;
            int pairs = features / 2;
            var ordered = b.Concat(1, b.Slice(reference, 1, 1, 1), b.Slice(reference, 1, 0, 1), b.Slice(reference, 1, 2, 2));
            var frequencies = new float[pairs];
            for (int k = 0; k < pairs; k++)
            {
                frequencies[k] = (float)(2 * Math.PI / Math.Pow(10000.0, 2.0 * k / features));
            }

            var angle = b.Mul(b.Reshape(b.Contiguous(ordered), _queries, 4, 1), b.Constant("sine_frequencies", DataType.F32, new Shape(1, 1, pairs), Bytes(frequencies)));
            var sin = b.Reshape(b.Unary("Sin", angle), _queries, 4, pairs, 1);
            var cos = b.Reshape(b.Unary("Cos", angle), _queries, 4, pairs, 1);
            return b.Reshape(b.Concat(3, sin, cos), _queries, 4 * features);
        }

        private Value Deformable(Value query, Value source, Value reference, string p)
        {
            var b = _b;
            int heads = _src.Arch<int>("decoder.cross_head_count");
            int points = _src.Arch<int>("decoder.point_count");
            int hd = _d / heads;
            int queries = _queries;
            var value = b.Linear(source, p + "value_proj.weight", p + "value_proj.bias");
            var offsets = b.Reshape(b.Linear(query, p + "sampling_offsets.weight", p + "sampling_offsets.bias"), queries, heads, 1, points, 2);
            var weights = b.Emit1("Softmax", [b.Reshape(b.Linear(query, p + "attention_weights.weight", p + "attention_weights.bias"), queries, heads, points)]);
            weights = b.Reshape(b.Contiguous(b.Permute(weights, 1, 0, 2)), heads, 1, queries, points);
            var center = b.Reshape(b.Slice(reference, 1, 0, 2), queries, 1, 1, 1, 2);
            var size = b.Reshape(b.Slice(reference, 1, 2, 2), queries, 1, 1, 1, 2);
            var location = b.Add(center, b.Mul(b.Scale(offsets, 0.5f / points), size));
            var grid = b.Scale(location, 2f, -1f);
            var level = b.Reshape(b.Contiguous(b.Permute(b.Reshape(value, _grid * _grid, heads, hd), 1, 2, 0)), heads, hd, _grid, _grid);
            var g = b.Contiguous(b.Permute(b.Reshape(grid, queries, heads, points, 2), 1, 0, 2, 3));
            var sampled = b.Emit1("GridSample", [level, g], Attributes.From(("mode", "linear"), ("padding", "zeros"), ("align_corners", false)));
            var weighted = b.Emit1("Reduce", [b.Mul(sampled, weights)], Attributes.From(("mode", "sum"), ("axes", new long[] { 3 }), ("keepdims", false)));
            var merged = b.Reshape(b.Contiguous(b.Permute(weighted, 2, 0, 1)), queries, _d);
            return b.Linear(merged, p + "output_proj.weight", p + "output_proj.bias");
        }

        private Value Attention(Value query, Value key, Value value, string p, int heads)
        {
            var b = _b;
            int hd = _d / heads;
            var q = b.Reshape(b.Linear(query, p + "q.weight", p + "q.bias"), _queries, heads, hd);
            var k = b.Reshape(b.Linear(key, p + "k.weight", p + "k.bias"), _queries, heads, hd);
            var v = b.Reshape(b.Linear(value, p + "v.weight", p + "v.bias"), _queries, heads, hd);
            var a = b.Emit1("Attention", [q, k, v], Attributes.From(("scale", 1f / MathF.Sqrt(hd)), ("causal", false)));
            return b.Linear(b.Reshape(a, _queries, _d), p + "o.weight", p + "o.bias");
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

        // Anchor boxes of every feature position: ((x + 0.5) / W, (y + 0.5) / H, 0.05, 0.05).
        private (List<float[]> Boxes, bool AllValid) Proposals()
        {
            var boxes = new List<float[]>();
            bool valid = true;
            for (int y = 0; y < _grid; y++)
            {
                for (int x = 0; x < _grid; x++)
                {
                    float[] a = [(x + 0.5f) / _grid, (y + 0.5f) / _grid, 0.05f, 0.05f];
                    valid &= a.All(v => v > 0.01f && v < 0.99f);
                    boxes.Add(a);
                }
            }

            if (!valid)
            {
                throw new NotSupportedException("rf_detr: feature positions whose anchors fall outside (0.01, 0.99) are not supported (the grid is too small).");
            }

            return (boxes, valid);
        }

        private static byte[] Bytes(float[] values) => MemoryMarshal.AsBytes(values.AsSpan()).ToArray();
    }
}
