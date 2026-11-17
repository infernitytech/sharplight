using System.Text.Json;
using System.Text.Json.Nodes;
using Sharplight.Graph;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// ConvNeXt image classifiers (canonical arch <c>convnext</c>: ConvNeXt V1 as published in timm and transformers): a 4x4 patchify stem with a layer norm, four stages of blocks (7x7 depthwise convolution, layer norm,
/// a point-wise MLP of ratio 4 with GELU, optional layer scale, residual) separated by layer-norm + 2x2 stride-2 convolution downsamplers, global average pooling, a final layer norm and a linear classifier.
/// Builds <see cref="WellKnown.ImageEncodeGraph"/> over the processed image as one row of <c>3 · size²</c> channel-major values and describes the model as <see cref="ModelKinds.ImageClassification"/>.
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>convnext.*</c>): <c>image_size</c>, <c>depths</c>, <c>hidden_sizes</c> (comma separated), <c>layer_norm_eps</c>; the canonical keys <see cref="CanonicalKeys.ClassifierLabels"/> (optional) and
/// <see cref="CanonicalKeys.ImageProcessor"/>. Weights: <c>stem.conv.*</c>, <c>stem.norm.*</c>, <c>stages.{s}.down.{norm,conv}.*</c> (s &gt; 0), <c>stages.{s}.blocks.{j}.{dw,norm,pw1,pw2}.*</c> and the optional
/// <c>stages.{s}.blocks.{j}.gamma.weight</c>, <c>head.norm.*</c>, <c>classifier.{weight, bias}</c>.
/// </remarks>
public sealed class ConvNextClassifierAdapter : IModelAdapter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "convnext";

    /// <inheritdoc/>
    public string Name => "ConvNeXt image classifier";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => [Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        int size = src.Arch<int>("image_size");
        int[] depths = [.. src.Arch<string>("depths").Split(',').Select(int.Parse)];
        int[] hidden = [.. src.Arch<string>("hidden_sizes").Split(',').Select(int.Parse)];
        float eps = src.Arch("layer_norm_eps", 1e-6f);

        var b = new GraphBuilder(WellKnown.ImageEncodeGraph, src.Weight);
        var input = b.Input("patches", InputRole.Media, DataType.F32, new Shape(1, 3L * size * size));
        var map = Conv(b, b.Reshape(input, 1, 3, size, size), "stem.conv", 4, 0, 1);
        int h = size / 4;
        var x = TowerBlocks.LayerNorm(b, ToRows(b, map, hidden[0], h * h), "stem.norm", eps);
        for (int s = 0; s < depths.Length; s++)
        {
            int c = hidden[s];
            if (s > 0)
            {
                var normed = TowerBlocks.LayerNorm(b, x, $"stages.{s}.down.norm", eps);
                map = Conv(b, ToMap(b, normed, hidden[s - 1], h, h), $"stages.{s}.down.conv", 2, 0, 1);
                h /= 2;
                x = ToRows(b, map, c, h * h);
            }

            for (int j = 0; j < depths[s]; j++)
            {
                string p = $"stages.{s}.blocks.{j}.";
                var dw = Conv(b, ToMap(b, x, c, h, h), p + "dw", 1, 3, c);
                var y = TowerBlocks.LayerNorm(b, ToRows(b, dw, c, h * h), p + "norm", eps);
                y = b.Linear(b.Unary("Gelu", b.Linear(y, p + "pw1.weight", p + "pw1.bias")), p + "pw2.weight", p + "pw2.bias");
                if (b.HasWeight(p + "gamma.weight"))
                {
                    y = b.Mul(y, b.Weight(p + "gamma.weight"));
                }

                x = b.Add(x, y);
            }
        }

        int last = hidden[^1];
        var pooled = b.Reshape(b.Emit1("Pool", [ToMap(b, x, last, h, h)], Attributes.From(("mode", "avg"), ("kernel", new long[] { h, h }), ("strides", new long[] { h, h }))), 1, last);
        b.Output(WellKnown.PooledOutput, pooled);
        var head = TowerBlocks.LayerNorm(b, pooled, "head.norm", eps);
        b.Output(WellKnown.LogitsOutput, b.Linear(head, "classifier.weight", "classifier.bias"));

        var labels = src.Get(CanonicalKeys.ClassifierLabels, Array.Empty<string>());
        var processor = JsonNode.Parse(src.Get<string>(CanonicalKeys.ImageProcessor))!;
        var description = new ModelDescription
        {
            Kind = ModelKinds.ImageClassification,
            Config = new Dictionary<string, JsonElement>
            {
                ["labels"] = JsonSerializer.SerializeToElement(labels),
                ["scoring"] = JsonSerializer.SerializeToElement("softmax"),
            },
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
                    Graphs = [WellKnown.ImageEncodeGraph],
                    Config = new Dictionary<string, JsonElement>
                    {
                        ["patchSize"] = JsonSerializer.SerializeToElement(size),
                        ["imageSize"] = JsonSerializer.SerializeToElement(size),
                    },
                },
            ],
        };
        return new ModelDefinition
        {
            Architecture = src.Architecture,
            Name = src.Get(CanonicalKeys.Name, src.Architecture),
            Weights = src.Weights,
            Graphs = new Dictionary<string, ComputeGraph> { [WellKnown.ImageEncodeGraph] = b.Build() },
            SourceMetadata = src.Metadata,
            OwnedResources = src.OwnedResources,
            Description = description,
        };
    }

    private static Value ToRows(GraphBuilder b, Value map, int channels, int rows)
        => b.Contiguous(b.Permute(b.Reshape(map, channels, rows), 1, 0));

    private static Value ToMap(GraphBuilder b, Value rows, int channels, int h, int w)
        => b.Reshape(b.Contiguous(b.Permute(rows, 1, 0)), 1, channels, h, w);

    private static Value Conv(GraphBuilder b, Value x, string name, int stride, int pad, int groups)
        => b.Emit1("Conv", [x, b.Weight(name + ".weight"), b.Weight(name + ".bias")],
            Attributes.From(("strides", new long[] { stride, stride }), ("pads", new long[] { pad, pad, pad, pad }), ("groups", (long)groups)));
}
