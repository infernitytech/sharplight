using System.Text.Json;
using System.Text.Json.Nodes;
using Sharplight.Graph;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Dense-prediction depth estimators on a vision transformer (canonical arch <c>dpt_depth</c>: Depth Anything, DPT): the backbone
/// (<see cref="ImageTowerGraph.Embed"/> + layer stack with intermediate taps), a reassemble stage that turns four tapped layers into feature maps of
/// different scales, a feature-fusion pyramid of residual convolution units and a convolutional head. Builds <see cref="WellKnown.DepthGraph"/> for a fixed
/// square input and describes the model as <see cref="ModelKinds.DepthEstimation"/>.
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>dpt_depth.*</c>): the backbone as <see cref="ImageEncoderAdapter"/> under <c>backbone.</c> (<c>layer_count</c>, <c>hidden_size</c>, ..., <c>patch_size</c>,
/// <c>image_size</c>), <c>tap_layers</c> ("3,6,9,12": 1-based layers), <c>neck.hidden_sizes</c> ("48,96,192,384"), <c>neck.factors</c> ("4,2,1,0.5": the scale of
/// each map relative to the patch grid), <c>fusion_hidden_size</c>, <c>head.hidden_size</c>. Weights: the backbone's under <c>backbone.</c>;
/// <c>neck.reassemble.{i}.{proj,resize}</c>, <c>neck.convs.{i}.weight</c> (no bias), <c>neck.fusion.{i}.{proj,res1.conv1,res1.conv2,res2.conv1,res2.conv2}</c>,
/// <c>head.{conv1,conv2,conv3}</c> (convolution weights [O, I, KH, KW], <c>resize</c> a transposed convolution for factors above 1).
/// </remarks>
public sealed class DepthEstimationAdapter : IModelAdapter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "dpt_depth";

    /// <inheritdoc/>
    public string Name => "Dense depth estimator";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => [Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        int patch = src.Arch<int>("backbone.patch_size");
        int size = src.Arch<int>("backbone.image_size");
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

    private static ComputeGraph BuildGraph(ModelSource src, int patch, int size)
    {
        var tower = new TowerOptions(src.Arch<int>("backbone.layer_count"), src.Arch<int>("backbone.hidden_size"), src.Arch<int>("backbone.attention.head_count"),
            src.Arch("backbone.norm_eps", 1e-6f), src.Arch("backbone.mlp.activation", "gelu"), "backbone.");
        var image = new ImageTowerOptions(tower, patch, size, src.Arch("backbone.register_count", 0), "cls");
        int[] taps = [.. src.Arch<string>("tap_layers").Split(',').Select(int.Parse)];
        int[] hidden = [.. src.Arch<string>("neck.hidden_sizes").Split(',').Select(int.Parse)];
        double[] factors = [.. src.Arch<string>("neck.factors").Split(',').Select(v => double.Parse(v, System.Globalization.CultureInfo.InvariantCulture))];
        int grid = size / patch;
        var b = new GraphBuilder(WellKnown.DepthGraph, src.Weight);
        var patches = b.Input("patches", InputRole.Media, DataType.F32, new Shape(grid * grid, 3 * patch * patch));
        var x = ImageTowerGraph.Embed(b, patches, image, out int rows);
        var tapped = new List<Value>();
        TowerBlocks.Layers(b, x, tower, rows, causal: false, tapped, taps);

        var features = new Value[taps.Length];
        var sizes = new (int H, int W)[taps.Length];
        for (int i = 0; i < taps.Length; i++)
        {
            using var _ = b.Group($"neck.reassemble.{i}");
            var normed = TowerBlocks.LayerNorm(b, tapped[i], "backbone.output_norm", tower.Eps);
            var map = b.Reshape(b.Contiguous(b.Permute(b.Slice(normed, 0, rows - grid * grid, grid * grid), 1, 0)), 1, tower.Hidden, grid, grid);
            map = Conv(b, map, $"neck.reassemble.{i}.proj", 1, 0);
            double f = factors[i];
            if (f > 1)
            {
                int k = (int)f;
                map = b.Emit1("ConvTranspose", [map, b.Weight($"neck.reassemble.{i}.resize.weight"), b.Weight($"neck.reassemble.{i}.resize.bias")],
                    Attributes.From(("strides", new long[] { k, k }), ("pads", new long[] { 0, 0, 0, 0 })));
            }
            else if (f < 1)
            {
                int stride = (int)Math.Round(1 / f);
                map = Conv(b, map, $"neck.reassemble.{i}.resize", stride, 1);
            }

            sizes[i] = f > 1 ? (grid * (int)f, grid * (int)f) : f < 1 ? ((grid + 2 - 3) / (int)Math.Round(1 / f) + 1, (grid + 2 - 3) / (int)Math.Round(1 / f) + 1) : (grid, grid);
            features[i] = b.Emit1("Conv", [map, b.Weight($"neck.convs.{i}.weight")], Attributes.From(("pads", new long[] { 1, 1, 1, 1 })));
        }

        // Fusion runs from the smallest map up; each stage resizes its result to the next (larger) map, the last one doubles it.
        Value? fused = null;
        var fusedSize = sizes[^1];
        int count = taps.Length;
        for (int idx = 0; idx < count; idx++)
        {
            using var _ = b.Group($"neck.fusion.{idx}");
            string p = $"neck.fusion.{idx}.";
            var level = features[count - 1 - idx];
            var current = level;
            if (fused is not null)
            {
                var residual = sizes[count - 1 - idx] == fusedSize ? level : Resize(b, level, fusedSize, alignCorners: false);
                current = b.Add(fused, ResidualUnit(b, residual, p + "res1."));
            }

            current = ResidualUnit(b, current, p + "res2.");
            fusedSize = idx + 1 < count ? sizes[count - 2 - idx] : (fusedSize.H * 2, fusedSize.W * 2);
            fused = Conv(b, Resize(b, current, fusedSize, alignCorners: true), p + "proj", 1, 0);
        }

        using (b.Group("head"))
        {
            var depth = Conv(b, fused!, "head.conv1", 1, 1);
            depth = Resize(b, depth, (grid * patch, grid * patch), alignCorners: true);
            depth = b.Unary("Relu", Conv(b, depth, "head.conv2", 1, 1));
            b.Output(WellKnown.DepthOutput, b.Unary("Relu", Conv(b, depth, "head.conv3", 1, 0)));
        }

        return b.Build();
    }

    private static Value ResidualUnit(GraphBuilder b, Value x, string p)
    {
        var y = Conv(b, b.Unary("Relu", x), p + "conv1", 1, 1);
        y = Conv(b, b.Unary("Relu", y), p + "conv2", 1, 1);
        return b.Add(y, x);
    }

    private static Value Conv(GraphBuilder b, Value x, string name, int stride, int pad)
        => b.Emit1("Conv", [x, b.Weight(name + ".weight"), b.Weight(name + ".bias")],
            Attributes.From(("strides", new long[] { stride, stride }), ("pads", new long[] { pad, pad, pad, pad })));

    private static Value Resize(GraphBuilder b, Value x, (int H, int W) size, bool alignCorners)
        => b.Emit1("Resize", [x], Attributes.From(("size", new long[] { size.H, size.W }), ("mode", "linear"), ("coord", alignCorners ? "align_corners" : "half_pixel")));
}
