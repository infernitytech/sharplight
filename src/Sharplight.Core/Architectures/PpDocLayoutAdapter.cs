using System.Text.Json;
using System.Text.Json.Nodes;
using Sharplight.Graph;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// PP-DocLayoutV3 document layout analysis (canonical arch <c>pp_doclayout_v3</c>): an HGNetV2 backbone, the RT-DETR hybrid encoder with a mask-feature pyramid
/// (stride-4 backbone features merged in, 32 prototype channels), top-k query selection whose initial boxes are enclosing boxes of the thresholded query masks (the
/// query's mask embedding dotted with the prototypes), the RT-DETR decoder with box heads shared across layers, and per-query mask, class and reading-order heads (a global
/// pointer over the queries). Builds <see cref="WellKnown.DetectGraph"/> for a fixed square input with the detection outputs <see cref="WellKnown.LogitsOutput"/> [Q, C]
/// and <see cref="WellKnown.BoxesOutput"/> [Q, 4], and <c>order_logits</c> [Q, Q] (the pairwise logits that query i comes before query j for j above i), <c>mask_embeddings</c>
/// [Q, P] and <c>mask_features</c> [P, h·w] (a query's mask logits are its embedding times the features).
/// </summary>
/// <remarks>
/// The shared hyper-parameters are the RT-DETR ones (<see cref="RtDetrAdapter"/>); further: <c>prototype_count</c>, <c>mask.channels</c> ("64,64"), <c>pointer.head_size</c>
/// and the backbone tables <c>backbone.stage_*</c> (comma lists, see <see cref="HgNetV2Options"/>). Weights: the RT-DETR names for the encoder, decoder and heads, with
/// <c>enc_box_head</c> and <c>enc_score_head</c> also the decoder's refinement and class heads; <c>backbone.*</c> per <see cref="HgNetV2Backbone"/>;
/// <c>mask_head.scale.{s}.{k}</c>, <c>mask_head.out</c>, <c>mask_lateral</c>, <c>mask_out.base</c>, <c>mask_out.proj</c> (convolutions with folded norms, the last with
/// a plain bias); <c>decoder_norm</c>, <c>mask_query.{0,1,2}</c>, <c>decoder.order_head.{i}</c>, <c>decoder.global_pointer</c>.
/// </remarks>
public sealed class PpDocLayoutAdapter : IModelAdapter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "pp_doclayout_v3";

    /// <summary>Component configuration key of the layout head's description (outputs of the detection graph).</summary>
    public const string HeadKey = "layoutHead";

    /// <inheritdoc/>
    public string Name => "PP-DocLayout document layout detector";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => [Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        var graph = new LayoutBuilder(src).Build();
        var labels = src.Get(CanonicalKeys.ClassifierLabels, Array.Empty<string>());
        var processor = JsonNode.Parse(src.Get<string>(CanonicalKeys.ImageProcessor))!;
        var head = new JsonObject
        {
            ["order"] = "order_logits",
            ["maskEmbeddings"] = "mask_embeddings",
            ["maskFeatures"] = "mask_features",
        };
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
                new ModelComponent
                {
                    Role = ModelComponentRoles.VisionEncoder,
                    Graphs = [WellKnown.DetectGraph],
                    Config = new Dictionary<string, JsonElement>
                    {
                        ["imageSize"] = JsonSerializer.SerializeToElement(src.Arch<int>("image_size")),
                        [HeadKey] = JsonSerializer.SerializeToElement(head),
                    },
                },
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

    private sealed class LayoutBuilder(ModelSource src) : RtDetrAdapter.Builder(src)
    {
        public override ComputeGraph Build()
        {
            var b = _b;
            var pixels = b.Input("pixels", InputRole.Media, DataType.F32, new Shape(1, 3, _size, _size));
            var stages = HgNetV2Backbone.Build(b, pixels, "backbone.", Tables());
            var maps = new Value[_levels];
            var shapes = new (int H, int W)[_levels];
            for (int i = 0; i < _levels; i++)
            {
                shapes[i] = (_size / _strides[i], _size / _strides[i]);
                maps[i] = ResNetBackbone.Conv(b, stages[stages.Length - _levels + i], $"encoder_proj.{i}", 1, 0, relu: false);
            }

            maps[_levels - 1] = Aifi(maps[_levels - 1], shapes[_levels - 1]);
            var pan = HybridEncoder(maps, shapes);
            var features = MaskFeatures(pan, stages[0]);
            int prototypes = _src.Arch<int>("prototype_count");
            int cells = _size / 4 * (_size / 4);
            var featureRows = b.Reshape(features, prototypes, cells);
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
                DecodeLayout(source, shapes, total, featureRows, _size / 4);
            }

            return b.Build();
        }

        private HgNetV2Options Tables()
        {
            int[] Ints(string key, string fallback) => [.. _src.Arch(key, fallback).Split(',').Select(int.Parse)];
            bool[] Bools(string key, string fallback) => [.. _src.Arch(key, fallback).Split(',').Select(v => v == "1")];
            return new HgNetV2Options(Ints("backbone.stem_channels", "3,32,48"), Ints("backbone.stage_in_channels", "48,128,512,1024"),
                Ints("backbone.stage_mid_channels", "48,96,192,384"), Ints("backbone.stage_out_channels", "128,512,1024,2048"), Ints("backbone.stage_blocks", "1,1,3,1"),
                Ints("backbone.stage_layers", "6,6,6,6"), Bools("backbone.stage_downsample", "0,1,1,1"), Bools("backbone.stage_light", "0,0,1,1"),
                Ints("backbone.stage_kernels", "3,3,5,5"));
        }

        // The pyramid over the three encoder outputs (strides 8, 16, 32: a convolution per level, summed at stride 8 after bilinear 2x upsampling per stride doubling),
        // upsampled to stride 4, plus the lateral convolution of the stride-4 backbone features, then the prototype projection. Returns [1, P, h, w].
        private Value MaskFeatures(Value[] pan, Value stride4)
        {
            var b = _b;
            Value sum = default!;
            for (int i = 0; i < pan.Length; i++)
            {
                var x = pan[i];
                int convs = Math.Max(1, (int)Math.Round(Math.Log2(_strides[i]) - Math.Log2(_strides[0])));
                int layer = 0;
                for (int k = 0; k < convs; k++)
                {
                    x = Silu(ResNetBackbone.Conv(b, x, $"mask_head.scale.{i}.{layer++}", 1, 1, relu: false));
                    if (_strides[i] != _strides[0])
                    {
                        x = Upsample(x, 2);
                        layer++;
                    }
                }

                sum = i == 0 ? x : b.Add(sum, x);
            }

            var features = Silu(ResNetBackbone.Conv(b, sum, "mask_head.out", 1, 1, relu: false));
            features = b.Add(Upsample(features, 2), Silu(ResNetBackbone.Conv(b, stride4, "mask_lateral", 1, 1, relu: false)));
            features = Silu(ResNetBackbone.Conv(b, features, "mask_out.base", 1, 1, relu: false));
            return ResNetBackbone.Conv(b, features, "mask_out.proj", 1, 0, relu: false);
        }

        private Value Silu(Value x) => _b.Silu(x);

        private Value Upsample(Value x, int factor)
        {
            long h = x.Shape[2].Fixed * factor;
            long w = x.Shape[3].Fixed * factor;
            return _b.Emit1("Resize", [x], Attributes.From(("size", new long[] { h, w }), ("mode", "linear"), ("coord", "half_pixel")));
        }

        private void DecodeLayout(Value source, (int H, int W)[] shapes, int total, Value featureRows, int maskSide)
        {
            var b = _b;
            var (_, valid) = Anchors(shapes);
            var memory = b.Mul(source, b.Constant("valid_mask", DataType.F32, new Shape(total, 1), Bytes(valid)));
            var outputMemory = LayerNorm(b.Linear(memory, "enc_output.linear.weight", "enc_output.linear.bias"), "enc_output.norm");
            var scores = b.Linear(outputMemory, "enc_score_head.weight", "enc_score_head.bias");
            int queries = _src.Arch<int>("query_count");
            var best = b.Reshape(b.Emit1("ReduceMax", [scores]), total);
            var top = b.Emit("TopK", [best], Attributes.From(("k", (long)queries)));
            var indices = top[1];
            var h = b.Gather(outputMemory, indices);
            // Initial boxes: the enclosing box of the pixels where the query's mask logit is positive.
            var initial = MaskBox(MaskLogits(LayerNorm(h, "decoder_norm"), featureRows), maskSide);
            var reference = b.Sigmoid(InverseSigmoid(initial));
            int layers = _src.Arch<int>("decoder.layer_count");
            string act = _src.Arch("decoder.mlp.activation", "relu");
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
                reference = b.Sigmoid(b.Add(Mlp(h, "enc_box_head.", 3), InverseSigmoid(reference)));
            }

            var outQuery = LayerNorm(h, "decoder_norm");
            b.Output(WellKnown.LogitsOutput, b.Linear(outQuery, "enc_score_head.weight", "enc_score_head.bias"));
            b.Output(WellKnown.BoxesOutput, reference);
            int head = _src.Arch<int>("pointer.head_size");
            var pair = b.Reshape(b.Linear(b.Linear(outQuery, $"decoder.order_head.{layers - 1}.weight", $"decoder.order_head.{layers - 1}.bias"),
                "decoder.global_pointer.weight", "decoder.global_pointer.bias"), queries, 2, head);
            var q = b.Reshape(b.Slice(pair, 1, 0, 1), queries, head);
            var k = b.Reshape(b.Slice(pair, 1, 1, 1), queries, head);
            var pointer = b.Emit1("BatchMatMul", [q, k], Attributes.From(("trans_b", true), ("alpha", 1f / MathF.Sqrt(head))));
            var upper = new float[queries * queries];
            var lower = new float[queries * queries];
            for (int i = 0; i < queries; i++)
            {
                for (int j = 0; j < queries; j++)
                {
                    (j > i ? upper : lower)[(i * queries) + j] = j > i ? 1f : -1e4f;
                }
            }

            var order = b.Add(b.Mul(pointer, b.Constant("order_upper", DataType.F32, new Shape(queries, queries), Bytes(upper))),
                b.Constant("order_lower", DataType.F32, new Shape(queries, queries), Bytes(lower)));
            b.Output("order_logits", order);
            b.Output("mask_embeddings", Mlp(outQuery, "mask_query.", 3));
            b.Output("mask_features", featureRows);
        }

        private Value MaskLogits(Value normedQuery, Value featureRows)
            => _b.Emit1("BatchMatMul", [Mlp(normedQuery, "mask_query.", 3), featureRows]);

        // mask_to_box_coordinate: (center x, center y, width, height) as fractions of the mask map for the pixels with a positive logit; zeros for an empty mask.
        private Value MaskBox(Value maskLogits, int side)
        {
            var b = _b;
            int cells = side * side;
            var xs = new float[cells];
            var ys = new float[cells];
            for (int i = 0; i < cells; i++)
            {
                xs[i] = i % side;
                ys[i] = i / side;
            }

            var mask = b.Binary("Greater", maskLogits, b.Constant("zero", DataType.F32, new Shape(1), new byte[4]));
            Value Max(Value v) => b.Emit1("Reduce", [v], Attributes.From(("mode", "max"), ("axes", new long[] { 1 }), ("keepdims", true)));
            Value Row(string name, float[] values) => b.Constant(name, DataType.F32, new Shape(1, cells), Bytes(values));
            var xMax = b.Scale(Max(b.Mul(mask, Row("mask_x", xs))), 1f, 1f);
            var yMax = b.Scale(Max(b.Mul(mask, Row("mask_y", ys))), 1f, 1f);
            var xMin = b.Scale(Max(b.Mul(mask, Row("mask_x_inverse", [.. xs.Select(x => side - x)]))), -1f, side);
            var yMin = b.Scale(Max(b.Mul(mask, Row("mask_y_inverse", [.. ys.Select(y => side - y)]))), -1f, side);
            var filled = Max(mask);
            var x0 = b.Mul(xMin, filled);
            var y0 = b.Mul(yMin, filled);
            var x1 = b.Mul(xMax, filled);
            var y1 = b.Mul(yMax, filled);
            float inverse = 1f / side;
            var center = b.Concat(1, b.Scale(b.Add(x0, x1), inverse / 2), b.Scale(b.Add(y0, y1), inverse / 2));
            var size = b.Concat(1, b.Scale(b.Sub(x1, x0), inverse), b.Scale(b.Sub(y1, y0), inverse));
            return b.Concat(1, center, size);
        }
    }
}
