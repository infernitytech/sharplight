using System.Text.Json;
using System.Text.Json.Nodes;
using Sharplight.Graph;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// BiRefNet / RMBG-2.0 dichotomous image segmentation (canonical arch <c>birefnet</c>): a Swin transformer encoder run on the image and on its half-size copy (the
/// features concatenated), a decoder of deformable-ASPP blocks with the image's own patches injected at every scale (<c>dec_ipt</c>) and gradient attention, producing a single-channel
/// matte. Builds <see cref="WellKnown.SemanticSegmentGraph"/> for a fixed square input (one class channel: the client reads it as a matte) and describes the model as
/// <see cref="ModelKinds.SemanticSegmentation"/>.
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>birefnet.*</c>): <c>image_size</c>, <c>embed_dim</c>, <c>depths</c>, <c>head_counts</c> (comma-separated), <c>window</c>, <c>decoder_inter</c> (the decoder blocks' inner width, 64),
/// <c>aspp_kernels</c> (the deformable branches' kernel sizes, 1,3,7); canonical key <see cref="CanonicalKeys.ImageProcessor"/>. Weights: <c>enc.*</c> (the Swin backbone: <c>patch_embed.proj</c>,
/// <c>patch_embed.norm</c>, <c>layers.{i}.blocks.{j}.*</c>, <c>layers.{i}.downsample.{norm, reduction}</c>, <c>norm{i}</c>), <c>squeeze.*</c> and <c>dec.*</c> (decoder blocks, lateral and
/// input-patch convolutions, gradient attention convolutions) with every batch norm folded into its convolution.
/// </remarks>
public sealed class BiRefNetAdapter : IModelAdapter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "birefnet";

    private const float NormEps = 1e-5f;

    /// <inheritdoc/>
    public string Name => "BiRefNet matting segmenter";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => [Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        int size = src.Arch<int>("image_size");
        var graph = new Builder(src, size).Build();
        var processor = JsonNode.Parse(src.Get<string>(CanonicalKeys.ImageProcessor))!;
        var description = new ModelDescription
        {
            Kind = ModelKinds.SemanticSegmentation,
            Components =
            [
                new ModelComponent
                {
                    Role = ModelComponentRoles.FeatureExtractor,
                    Config = new Dictionary<string, JsonElement> { [ImageProcessor.ComponentKey] = JsonSerializer.SerializeToElement(processor) },
                },
                new ModelComponent { Role = ModelComponentRoles.VisionEncoder, Graphs = [WellKnown.SemanticSegmentGraph], Config = ComponentConfig.Of(("imageSize", size)) },
            ],
        };
        return new ModelDefinition
        {
            Architecture = src.Architecture,
            Name = src.Get(CanonicalKeys.Name, src.Architecture),
            Weights = src.Weights,
            Graphs = new Dictionary<string, ComputeGraph> { [WellKnown.SemanticSegmentGraph] = graph },
            SourceMetadata = src.Metadata,
            OwnedResources = src.OwnedResources,
            Description = description,
        };
    }

    private sealed class Builder
    {
        private readonly int _size;
        private readonly GraphBuilder _b;
        private readonly ConstantCache _k;
        private readonly int _embed;
        private readonly int[] _depths;
        private readonly int[] _heads;
        private readonly int _window;
        private readonly int[] _asppKernels;

        public Builder(ModelSource src, int size)
        {
            _size = size;
            _b = new GraphBuilder(WellKnown.SemanticSegmentGraph, src.Weight);
            _k = new ConstantCache(_b);
            _embed = src.Arch<int>("embed_dim");
            _depths = [.. src.Arch<string>("depths").Split(',').Select(int.Parse)];
            _heads = [.. src.Arch<string>("head_counts").Split(',').Select(int.Parse)];
            _window = src.Arch<int>("window");
            _asppKernels = [.. src.Arch<string>("aspp_kernels").Split(',').Select(int.Parse)];
        }

        public ComputeGraph Build()
        {
            var pixels = _b.Input("pixels", InputRole.Media, DataType.F32, new Shape(1, 3, _size, _size));
            var full = Backbone(pixels, _size);
            var half = Backbone(Resize(pixels, _size / 2, _size / 2), _size / 2);
            var x = new Value[4];
            for (int i = 0; i < 4; i++)
            {
                x[i] = _b.Concat(1, full[i], Resize(half[i], (int)full[i].Shape[2].Fixed, (int)full[i].Shape[3].Fixed));
            }

            int h4 = (int)x[3].Shape[2].Fixed;
            int w4 = (int)x[3].Shape[3].Fixed;
            var deep = _b.Concat(1, Resize(x[0], h4, w4), Resize(x[1], h4, w4), Resize(x[2], h4, w4), x[3]);
            deep = DecoderBlock(deep, "squeeze");
            var logits = Decode(pixels, x[0], x[1], x[2], deep);
            _b.Output(WellKnown.SemanticLogitsOutput, logits);
            return _b.Build();
        }

        // The Swin backbone: four feature maps [1, C·2^i, H/(4·2^i), W/(4·2^i)] after the stage norms.
        private Value[] Backbone(Value pixels, int size)
        {
            var maps = new Value[4];
            int h = size / 4;
            int w = size / 4;
            int c = _embed;
            var embedded = Conv(pixels, "enc.patch_embed.proj", 4, 0, 1);
            var x = LayerNorm(Rows(embedded, c, h, w), "enc.patch_embed.norm");
            for (int s = 0; s < 4; s++)
            {
                using var stage = _b.Group($"enc.layers.{s}");
                for (int j = 0; j < _depths[s]; j++)
                {
                    x = Block(x, $"enc.layers.{s}.blocks.{j}.", c, _heads[s], h, w, j % 2 == 1 ? _window / 2 : 0);
                }

                maps[s] = Map(LayerNorm(x, $"enc.norm{s}"), c, h, w);
                if (s < 3)
                {
                    x = Merge(x, $"enc.layers.{s}.downsample.", h, w, c);
                    h /= 2;
                    w /= 2;
                    c *= 2;
                }
            }

            return maps;
        }

        // One Swin block over rows x [H·W, C]: pre-norm window attention over zero-padded windows (shifted on odd blocks) and a pre-norm MLP.
        private Value Block(Value x, string p, int c, int heads, int height, int width, int shift)
        {
            int d = c / heads;
            int window = _window;
            int n = window * window;
            int hp = SwinWindows.Padded(height, window);
            int wp = SwinWindows.Padded(width, window);
            int windows = hp / window * (wp / window);
            long batch = (long)windows * heads;

            var h = LayerNorm(x, p + "norm1");
            var (into, back) = SwinWindows.Indices(height, width, window, shift);
            if (hp != height || wp != width)
            {
                h = _b.Concat(0, h, _k.F32($"zero_row_{c}", new Shape(1, c), () => new float[c]));
            }

            h = _b.Gather(h, _k.I32($"win_{height}_{width}_{shift}", () => into));
            var qkv = _b.Linear(h, p + "attn.qkv.weight", p + "attn.qkv.bias");
            Value Heads(int part) => _b.Reshape(_b.Permute(_b.Reshape(_b.Slice(qkv, 1, (long)part * c, c), windows, n, heads, d), 0, 2, 1, 3), batch, n, d);
            var scores = _b.Emit1("BatchMatMul", [Heads(0), Heads(1)], Attributes.From(("trans_b", true), ("alpha", 1f / MathF.Sqrt(d))));
            scores = _b.Reshape(scores, windows, heads, n, n);
            var table = _b.Gather(_b.Weight(p + "attn.relative_position_bias_table"), _k.I32($"rel_index_{window}", () => SwinWindows.RelativeIndex(window)));
            scores = _b.Add(scores, _b.Permute(_b.Reshape(table, n, n, heads), 2, 0, 1));
            if (shift > 0)
            {
                scores = _b.Add(scores, _k.F32($"mask_{hp}_{wp}_{window}", new Shape(windows, 1, n, n), () => SwinWindows.ShiftMask(hp, wp, window, shift)));
            }

            var probs = _b.Emit1("Softmax", [_b.Reshape(scores, batch, n, n)]);
            var o = _b.Emit1("BatchMatMul", [probs, Heads(2)]);
            o = _b.Reshape(_b.Contiguous(_b.Permute(_b.Reshape(o, windows, heads, n, d), 0, 2, 1, 3)), (long)windows * n, c);
            o = _b.Linear(o, p + "attn.proj.weight", p + "attn.proj.bias");
            o = _b.Gather(o, _k.I32($"unwin_{height}_{width}_{shift}", () => back));
            x = _b.Add(x, o);
            var m = _b.Linear(_b.Unary("Gelu", _b.Linear(LayerNorm(x, p + "norm2"), p + "mlp.fc1.weight", p + "mlp.fc1.bias")), p + "mlp.fc2.weight", p + "mlp.fc2.bias");
            return _b.Add(x, m);
        }

        // Patch merging: the four cells of every 2 x 2 neighbourhood side by side (column-major order of the reference), normalized and reduced to twice the channels.
        private Value Merge(Value x, string p, int height, int width, int c)
        {
            long hh = height / 2;
            long hw = width / 2;
            var grid = _b.Reshape(x, hh, 2, hw, 2, c);
            var parts = new Value[4];
            for (int col = 0; col < 2; col++)
            {
                for (int row = 0; row < 2; row++)
                {
                    parts[(col * 2) + row] = _b.Reshape(_b.Slice(_b.Slice(grid, 1, row, 1), 3, col, 1), hh * hw, c);
                }
            }

            return _b.Linear(LayerNorm(_b.Concat(1, parts), p + "norm"), p + "reduction.weight");
        }

        private Value Decode(Value pixels, Value x1, Value x2, Value x3, Value x4)
        {
            int[] side = [(int)x4.Shape[2].Fixed, (int)x3.Shape[2].Fixed, (int)x2.Shape[2].Fixed, (int)x1.Shape[2].Fixed];
            var p4 = Gradient(DecoderBlock(_b.Concat(1, x4, Patches(pixels, side[0], "dec.ipt_blk5")), "dec.decoder_block4"), 4);
            var p3 = Gradient(DecoderBlock(_b.Concat(1, Merged(p4, x3, 4, side[1]), Patches(pixels, side[1], "dec.ipt_blk4")), "dec.decoder_block3"), 3);
            var p2 = Gradient(DecoderBlock(_b.Concat(1, Merged(p3, x2, 3, side[2]), Patches(pixels, side[2], "dec.ipt_blk3")), "dec.decoder_block2"), 2);
            var p1 = DecoderBlock(_b.Concat(1, Merged(p2, x1, 2, side[3]), Patches(pixels, side[3], "dec.ipt_blk2")), "dec.decoder_block1");
            p1 = Resize(p1, _size, _size);
            p1 = _b.Concat(1, p1, Patches(pixels, _size, "dec.ipt_blk1"));
            return Conv(p1, "dec.conv_out1.0", 1, 0, 1);
        }

        // The upsampled coarser prediction plus the lateral convolution of the encoder feature.
        private Value Merged(Value coarse, Value lateral, int block, int side)
            => _b.Add(Resize(coarse, side, side), Conv(lateral, $"dec.lateral_block{block}.conv", 1, 0, 1));

        // Gradient attention: a 16-channel conv + ReLU of the block output predicts a one-channel attention that scales it.
        private Value Gradient(Value p, int level)
        {
            var g = _b.Unary("Relu", Conv(p, $"dec.gdt_convs_{level}.0", 1, 1, 1));
            return _b.Mul(p, _b.Sigmoid(Conv(g, $"dec.gdt_convs_attn_{level}.0", 1, 0, 1)));
        }

        // The image cut into a grid of tiles of the stage's side, one channel per (color, tile) pair, through two convolutions.
        private Value Patches(Value pixels, int side, string name)
        {
            int grid = _size / side;
            Value tiles = pixels;
            if (grid > 1)
            {
                var split = _b.Reshape(pixels, 3, grid, side, grid, side);
                tiles = _b.Reshape(_b.Contiguous(_b.Permute(split, 0, 1, 3, 2, 4)), 1, 3L * grid * grid, side, side);
            }

            return Conv(Conv(tiles, name + ".conv1", 1, 1, 1), name + ".conv_out", 1, 1, 1);
        }

        // conv 3x3 -> ReLU -> deformable ASPP -> conv 3x3 (the batch norms are folded into the convolutions).
        private Value DecoderBlock(Value x, string name)
        {
            var h = _b.Unary("Relu", Conv(x, name + ".conv_in", 1, 1, 1));
            h = Aspp(h, name + ".dec_att.");
            return Conv(h, name + ".conv_out", 1, 1, 1);
        }

        private Value Aspp(Value x, string p)
        {
            int height = (int)x.Shape[2].Fixed;
            int width = (int)x.Shape[3].Fixed;
            var branches = new List<Value> { _b.Unary("Relu", Deformable(x, p + "aspp1.atrous_conv.")) };
            for (int i = 0; i < _asppKernels.Length; i++)
            {
                branches.Add(_b.Unary("Relu", Deformable(x, $"{p}aspp_deforms.{i}.atrous_conv.")));
            }

            var pooled = _b.Emit1("Reduce", [x], Attributes.From(("mode", "mean"), ("axes", new long[] { 2, 3 }), ("keepdims", true)));
            pooled = _b.Unary("Relu", Conv(pooled, p + "global_avg_pool.1", 1, 0, 1));
            branches.Add(_b.Emit1("Expand", [pooled], Attributes.From(("shape", new long[] { 1, pooled.Shape[1].Fixed, height, width }))));
            return _b.Unary("Relu", Conv(_b.Concat(1, [.. branches]), p + "conv1", 1, 0, 1));
        }

        // Modulated deformable convolution: offsets and the mask (2 * sigmoid) come from convolutions of the input; the regular weight carries the folded batch norm.
        private Value Deformable(Value x, string p)
        {
            var weight = _b.Weight(p + "regular_conv.weight");
            int kernel = (int)weight.Shape[2].Fixed;
            int pad = kernel / 2;
            var offset = Conv(x, p + "offset_conv", 1, pad, 1);
            var mask = _b.Scale(_b.Sigmoid(Conv(x, p + "modulator_conv", 1, pad, 1)), 2f);
            return _b.Emit1("DeformConv", [x, weight, offset, _b.Weight(p + "regular_conv.bias"), mask],
                Attributes.From(("strides", new long[] { 1, 1 }), ("pads", new long[] { pad, pad, pad, pad }), ("dilations", new long[] { 1, 1 }), ("groups", 1L), ("offset_groups", 1L)));
        }

        // Bilinear resize with align_corners = True (the reference's F.interpolate).
        private Value Resize(Value x, int height, int width)
            => x.Shape[2].Fixed == height && x.Shape[3].Fixed == width
                ? x
                : _b.Emit1("Resize", [x], Attributes.From(("size", new long[] { height, width }), ("mode", "linear"), ("coord", "align_corners")));

        private Value Conv(Value x, string name, int stride, int pad, int groups)
        {
            var inputs = _b.HasWeight(name + ".bias") ? new[] { _b.Contiguous(x), _b.Weight(name + ".weight"), _b.Weight(name + ".bias") } : [_b.Contiguous(x), _b.Weight(name + ".weight")];
            return _b.Emit1("Conv", inputs, Attributes.From(("strides", new long[] { stride, stride }), ("pads", new long[] { pad, pad, pad, pad }), ("groups", (long)groups)));
        }

        private Value LayerNorm(Value x, string name)
            => _b.Emit1("LayerNorm", [x, _b.Weight(name + ".weight"), _b.Weight(name + ".bias")], Attributes.From(("eps", NormEps)));

        private Value Rows(Value map, int channels, int height, int width)
            => _b.Reshape(_b.Contiguous(_b.Permute(map, 0, 2, 3, 1)), (long)height * width, channels);

        private Value Map(Value rows, int channels, int height, int width)
            => _b.Contiguous(_b.Permute(_b.Reshape(rows, 1, height, width, channels), 0, 3, 1, 2));
    }
}
