using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Residual-in-residual dense block super-resolution networks (canonical arch <c>rrdbnet</c>: ESRGAN, Real-ESRGAN x4plus / x2plus): a convolution, a trunk of
/// residual-in-residual dense blocks (three dense blocks of five convolutions with growth channels, residual scale 0.2) with a long skip, two nearest-neighbour
/// 2× upsampling stages with LeakyReLU convolutions and a high-resolution convolution pair. Builds <see cref="WellKnown.UpscaleGraph"/> for the input size of
/// <see cref="AdapterOptions.Dimensions"/> (default <c>rrdbnet.default_size</c>) and describes the model as <see cref="ModelKinds.ImageUpscaling"/>.
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>rrdbnet.*</c>): <c>block_count</c> (23), <c>scale</c> (the net scale: 4, or 2 and 1 for
/// models that pixel-unshuffle their input by 4 / 2 / 1 first, as ESRGAN's x2 and x1 variants), <c>default_size</c> (the input side of the default graph, default 64).
/// Weights keep the names of the reference implementation: <c>conv_first</c>, <c>body.{i}.rdb{1,2,3}.conv{1..5}</c>, <c>conv_body</c>, <c>conv_up1</c>,
/// <c>conv_up2</c>, <c>conv_hr</c>, <c>conv_last</c> (each <c>.weight</c> [O, I, 3, 3] and <c>.bias</c>).
/// </remarks>
public sealed class RrdbNetAdapter : IModelAdapter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "rrdbnet";

    private const float Slope = 0.2f;
    private const float ResidualScale = 0.2f;

    /// <inheritdoc/>
    public string Name => "Residual-in-residual dense block upscaler";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => [Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        int blocks = src.Arch<int>("block_count");
        int scale = src.Arch("scale", 4);
        int unshuffle = 4 / scale;
        long height = options.Dimensions.GetValueOrDefault(WellKnown.UpscaleHeightDimension, src.Arch("default_size", 64));
        long width = options.Dimensions.GetValueOrDefault(WellKnown.UpscaleWidthDimension, src.Arch("default_size", 64));
        if (height % unshuffle != 0 || width % unshuffle != 0)
        {
            throw new ArgumentException($"rrdbnet: the input size {width}x{height} must be a multiple of {unshuffle}.");
        }

        var b = new GraphBuilder(WellKnown.UpscaleGraph, src.Weight);
        var pixels = b.Input("pixels", InputRole.Media, DataType.F32, new Shape(1, 3, height, width));
        var x = unshuffle > 1 ? PixelUnshuffle(b, pixels, unshuffle, height, width) : pixels;
        var first = Conv(b, x, "conv_first");
        var trunk = first;
        for (int i = 0; i < blocks; i++)
        {
            using var _ = b.Group($"body.{i}");
            var block = trunk;
            for (int r = 1; r <= 3; r++)
            {
                block = DenseBlock(b, block, $"body.{i}.rdb{r}.");
            }

            trunk = b.Add(b.Scale(block, ResidualScale), trunk);
        }

        var feat = b.Add(first, Conv(b, trunk, "conv_body"));
        using (b.Group("upsample"))
        {
            long h = height / unshuffle;
            long w = width / unshuffle;
            feat = LeakyRelu(b, Conv(b, Upsample(b, feat, h * 2, w * 2), "conv_up1"));
            feat = LeakyRelu(b, Conv(b, Upsample(b, feat, h * 4, w * 4), "conv_up2"));
            b.Output(WellKnown.UpscaledOutput, Conv(b, LeakyRelu(b, Conv(b, feat, "conv_hr")), "conv_last"));
        }

        var graph = b.Build();
        var description = new ModelDescription
        {
            Kind = ModelKinds.ImageUpscaling,
            Components =
            [
                new ModelComponent
                {
                    Role = ModelComponentRoles.VisionEncoder,
                    Graphs = [WellKnown.UpscaleGraph],
                    Config = ComponentConfig.Of(("scale", scale), ("height", (int)height), ("width", (int)width), ("multiple", unshuffle)),
                },
            ],
        };
        return new ModelDefinition
        {
            Architecture = src.Architecture,
            Name = src.Get(CanonicalKeys.Name, src.Architecture),
            Weights = src.Weights,
            Graphs = new Dictionary<string, ComputeGraph> { [WellKnown.UpscaleGraph] = graph },
            SourceMetadata = src.Metadata,
            OwnedResources = src.OwnedResources,
            Description = description,
        };
    }

    // x [N, C, H, W] → [N, C·r², H/r, W/r] with PyTorch's channel order (c, i, j).
    private static Value PixelUnshuffle(GraphBuilder b, Value x, int r, long height, long width)
    {
        var split = b.Reshape(x, 1, 3, height / r, r, width / r, r);
        return b.Reshape(b.Contiguous(b.Permute(split, 0, 1, 3, 5, 2, 4)), 1, 3L * r * r, height / r, width / r);
    }

    private static Value DenseBlock(GraphBuilder b, Value x, string p)
    {
        var parts = new List<Value> { x };
        for (int i = 1; i <= 4; i++)
        {
            var input = parts.Count == 1 ? x : b.Concat(1, [.. parts]);
            parts.Add(LeakyRelu(b, Conv(b, input, $"{p}conv{i}")));
        }

        var last = Conv(b, b.Concat(1, [.. parts]), $"{p}conv5");
        return b.Add(b.Scale(last, ResidualScale), x);
    }

    private static Value Conv(GraphBuilder b, Value x, string name)
        => b.Emit1("Conv", [x, b.Weight(name + ".weight"), b.Weight(name + ".bias")], Attributes.From(("strides", new long[] { 1, 1 }), ("pads", new long[] { 1, 1, 1, 1 })));

    private static Value LeakyRelu(GraphBuilder b, Value x) => b.Emit1("LeakyRelu", [x], Attributes.From(("alpha", Slope)));

    private static Value Upsample(GraphBuilder b, Value x, long height, long width)
        => b.Emit1("Resize", [x], Attributes.From(("size", new long[] { height, width }), ("mode", "nearest"), ("coord", "asymmetric"), ("nearest_mode", "floor")));
}
