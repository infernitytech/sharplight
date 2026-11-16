using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Swin transformer image restoration networks (canonical arch <c>swinir</c>: SwinIR classical and real-world super-resolution): a 3x3 convolution, residual Swin
/// transformer groups (pre-norm blocks of window attention with a learned relative position bias, alternating shifted windows and an additive mask, GELU MLP, a
/// 3x3 convolution and a skip per group), a final LayerNorm, a convolution with the long skip and an upsampler, either sub-pixel convolutions (classical) or
/// nearest-neighbour stages with convolutions (<c>nearest+conv</c>, real-world x4). Builds <see cref="WellKnown.UpscaleGraph"/> for the input size of
/// <see cref="AdapterOptions.Dimensions"/> (multiples of the window; default <c>swinir.default_size</c>) and describes the model as <see cref="ModelKinds.ImageUpscaling"/>.
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>swinir.*</c>): <c>window_size</c> (8), <c>scale</c>, <c>upsampler</c> (<c>pixelshuffle</c> or <c>nearest+conv</c>), <c>img_range</c> (1), <c>default_size</c>
/// (64). Group, block and head counts and the MLP width follow from the weights, which keep the names of the reference implementation
/// (<c>conv_first</c>, <c>patch_embed.norm</c>, <c>layers.{i}.residual_group.blocks.{j}.{norm1, attn.qkv, attn.proj, attn.relative_position_bias_table, norm2, mlp.fc1, mlp.fc2}</c>,
/// <c>layers.{i}.conv</c>, <c>norm</c>, <c>conv_after_body</c>, <c>conv_before_upsample.0</c>, <c>upsample.{0,2,..}</c> or <c>conv_up1</c>, <c>conv_up2</c>, <c>conv_hr</c>, <c>conv_last</c>).
/// The relative position index and the shift masks are rebuilt for the graph's size.
/// </remarks>
public sealed class SwinIrAdapter : IModelAdapter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "swinir";

    /// <summary>The upsampler of the classical models: sub-pixel convolutions.</summary>
    public const string PixelShuffleUpsampler = "pixelshuffle";

    /// <summary>The upsampler of the real-world x4 models: nearest-neighbour stages with convolutions.</summary>
    public const string NearestConvUpsampler = "nearest+conv";

    private const float NormEps = 1e-5f;
    private static readonly float[] s_rgbMean = [0.4488f, 0.4371f, 0.4040f];

    /// <inheritdoc/>
    public string Name => "Swin transformer image restoration";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => [Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        int window = src.Arch("window_size", 8);
        int scale = src.Arch<int>("scale");
        string upsampler = src.Arch("upsampler", PixelShuffleUpsampler);
        float imgRange = src.Arch("img_range", 1f);
        long height = options.Dimensions.GetValueOrDefault(WellKnown.UpscaleHeightDimension, src.Arch("default_size", 64));
        long width = options.Dimensions.GetValueOrDefault(WellKnown.UpscaleWidthDimension, src.Arch("default_size", 64));
        if (height % window != 0 || width % window != 0)
        {
            throw new ArgumentException($"swinir: the input size {width}x{height} must be a multiple of the window size {window}.");
        }

        int h = (int)height;
        int w = (int)width;
        var b = new GraphBuilder(WellKnown.UpscaleGraph, src.Weight);
        var k = new ConstantCache(b);
        int groups = Count(src, i => $"layers.{i}.conv.weight");
        int embed = (int)src.Weights["conv_first.weight"].Shape[0].Fixed;
        var mean = k.F32("rgb_mean", new Shape(1, 3, 1, 1), () => s_rgbMean);
        var pixels = b.Input("pixels", InputRole.Media, DataType.F32, new Shape(1, 3, height, width));
        var x = b.Scale(b.Sub(pixels, mean), imgRange);
        var first = Conv(b, x, "conv_first");

        // Patch embedding: the map as rows [H·W, C] with a LayerNorm.
        var tokens = LayerNorm(b, Rows(b, first, embed, h, w), "patch_embed.norm");
        for (int i = 0; i < groups; i++)
        {
            using var _ = b.Group($"layers.{i}");
            var group = tokens;
            int blocks = Count(src, j => $"layers.{i}.residual_group.blocks.{j}.norm1.weight");
            for (int j = 0; j < blocks; j++)
            {
                group = Block(b, k, src, group, i, j, window, h, w);
            }

            var map = Conv(b, Map(b, group, embed, h, w), $"layers.{i}.conv");
            tokens = b.Add(Rows(b, map, embed, h, w), tokens);
        }

        var body = Conv(b, Map(b, LayerNorm(b, tokens, "norm"), embed, h, w), "conv_after_body");
        var feat = b.Add(body, first);
        using (b.Group("upsample"))
        {
            Value result;
            if (upsampler == PixelShuffleUpsampler)
            {
                feat = Conv(b, feat, "conv_before_upsample.0");
                feat = b.Emit1("LeakyRelu", [feat], Attributes.From(("alpha", 0.01f)));
                int features = (int)src.Weights["conv_before_upsample.0.weight"].Shape[0].Fixed;
                int stages = Count(src, i => $"upsample.{2 * i}.weight");
                int factor = scale == 3 ? 3 : 2;
                long uh = height;
                long uw = width;
                for (int s = 0; s < stages; s++)
                {
                    feat = PixelShuffle(b, Conv(b, feat, $"upsample.{2 * s}"), features, factor, uh, uw);
                    uh *= factor;
                    uw *= factor;
                }

                result = Conv(b, feat, "conv_last");
            }
            else
            {
                feat = b.Emit1("LeakyRelu", [Conv(b, feat, "conv_before_upsample.0")], Attributes.From(("alpha", 0.01f)));
                feat = LeakyRelu(b, Conv(b, Nearest(b, feat, height * 2, width * 2), "conv_up1"));
                feat = LeakyRelu(b, Conv(b, Nearest(b, feat, height * 4, width * 4), "conv_up2"));
                result = Conv(b, LeakyRelu(b, Conv(b, feat, "conv_hr")), "conv_last");
            }

            b.Output(WellKnown.UpscaledOutput, b.Add(b.Scale(result, 1f / imgRange), mean));
        }

        var description = new ModelDescription
        {
            Kind = ModelKinds.ImageUpscaling,
            Components =
            [
                new ModelComponent
                {
                    Role = ModelComponentRoles.VisionEncoder,
                    Graphs = [WellKnown.UpscaleGraph],
                    Config = ComponentConfig.Of(("scale", scale), ("height", h), ("width", w), ("multiple", window)),
                },
            ],
        };
        return new ModelDefinition
        {
            Architecture = src.Architecture,
            Name = src.Get(CanonicalKeys.Name, src.Architecture),
            Weights = src.Weights,
            Graphs = new Dictionary<string, ComputeGraph> { [WellKnown.UpscaleGraph] = b.Build() },
            SourceMetadata = src.Metadata,
            OwnedResources = src.OwnedResources,
            Description = description,
        };
    }

    // The number of consecutive indices i = 0, 1, ... for which the weight name(i) exists.
    private static int Count(ModelSource src, Func<int, string> name)
    {
        int n = 0;
        while (src.Weights.ContainsKey(name(n)))
        {
            n++;
        }

        return n;
    }

    /// <summary>One Swin block over rows x [H·W, C]: pre-norm window attention (shifted on odd blocks) and a pre-norm MLP, both with residuals.</summary>
    private static Value Block(GraphBuilder b, ConstantCache k, ModelSource src, Value x, int group, int block, int window, int height, int width)
    {
        string p = $"layers.{group}.residual_group.blocks.{block}.";
        int c = (int)src.Weights[p + "norm1.weight"].Shape[0].Fixed;
        int heads = (int)src.Weights[p + "attn.relative_position_bias_table"].Shape[1].Fixed;
        int d = c / heads;
        int shift = block % 2 == 1 ? window / 2 : 0;
        int n = window * window;
        int windows = height / window * (width / window);
        long batch = (long)windows * heads;

        var h = LayerNorm(b, x, p + "norm1");
        var (into, back) = SwinWindows.Indices(height, width, window, shift);
        h = b.Gather(h, k.I32($"win_{height}_{width}_{shift}", () => into));                                         // [windows·N, C]
        var qkv = b.Linear(h, p + "attn.qkv.weight", p + "attn.qkv.bias");
        Value Heads(int part) => b.Reshape(b.Permute(b.Reshape(b.Slice(qkv, 1, (long)part * c, c), windows, n, heads, d), 0, 2, 1, 3), batch, n, d);
        var scores = b.Emit1("BatchMatMul", [Heads(0), Heads(1)], Attributes.From(("trans_b", true), ("alpha", 1f / MathF.Sqrt(d))));
        scores = b.Reshape(scores, windows, heads, n, n);
        var table = b.Gather(b.Weight(p + "attn.relative_position_bias_table"), k.I32($"rel_index_{window}", () => SwinWindows.RelativeIndex(window)));
        var bias = b.Permute(b.Reshape(table, n, n, heads), 2, 0, 1);                                                // [heads, N, N]
        scores = b.Add(scores, bias);
        if (shift > 0)
        {
            scores = b.Add(scores, k.F32($"mask_{height}_{width}_{window}", new Shape(windows, 1, n, n), () => SwinWindows.ShiftMask(height, width, window, shift)));
        }

        var probs = b.Emit1("Softmax", [b.Reshape(scores, batch, n, n)]);
        var o = b.Emit1("BatchMatMul", [probs, Heads(2)]);                                                           // [batch, N, d]
        o = b.Reshape(b.Contiguous(b.Permute(b.Reshape(o, windows, heads, n, d), 0, 2, 1, 3)), (long)windows * n, c);
        o = b.Linear(o, p + "attn.proj.weight", p + "attn.proj.bias");
        o = b.Gather(o, k.I32($"unwin_{height}_{width}_{shift}", () => back));
        x = b.Add(x, o);
        var m = b.Linear(b.Unary("Gelu", b.Linear(LayerNorm(b, x, p + "norm2"), p + "mlp.fc1.weight", p + "mlp.fc1.bias")), p + "mlp.fc2.weight", p + "mlp.fc2.bias");
        return b.Add(x, m);
    }

    // [1, C, H, W] as rows [H·W, C].
    private static Value Rows(GraphBuilder b, Value map, int channels, int height, int width)
        => b.Reshape(b.Contiguous(b.Permute(map, 0, 2, 3, 1)), (long)height * width, channels);

    // Rows [H·W, C] as [1, C, H, W].
    private static Value Map(GraphBuilder b, Value rows, int channels, int height, int width)
        => b.Contiguous(b.Permute(b.Reshape(rows, 1, height, width, channels), 0, 3, 1, 2));

    // PixelShuffle: [1, C·r², H, W] → [1, C, H·r, W·r] with channel order (c, i, j).
    private static Value PixelShuffle(GraphBuilder b, Value x, int channels, int r, long height, long width)
    {
        var split = b.Reshape(x, 1, channels, r, r, height, width);
        return b.Reshape(b.Contiguous(b.Permute(split, 0, 1, 4, 2, 5, 3)), 1, channels, height * r, width * r);
    }

    private static Value Conv(GraphBuilder b, Value x, string name)
    {
        var weight = b.Weight(name + ".weight");
        long kernel = weight.Shape[2].Fixed;
        long pad = kernel / 2;
        return b.Emit1("Conv", [b.Contiguous(x), weight, b.Weight(name + ".bias")], Attributes.From(("strides", new long[] { 1, 1 }), ("pads", new long[] { pad, pad, pad, pad })));
    }

    private static Value LayerNorm(GraphBuilder b, Value x, string name)
        => b.Emit1("LayerNorm", [x, b.Weight(name + ".weight"), b.Weight(name + ".bias")], Attributes.From(("eps", NormEps)));

    private static Value LeakyRelu(GraphBuilder b, Value x) => b.Emit1("LeakyRelu", [x], Attributes.From(("alpha", 0.2f)));

    private static Value Nearest(GraphBuilder b, Value x, long height, long width)
        => b.Emit1("Resize", [x], Attributes.From(("size", new long[] { height, width }), ("mode", "nearest"), ("coord", "asymmetric"), ("nearest_mode", "floor")));
}
