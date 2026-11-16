using System.Text.Json;
using System.Text.Json.Nodes;
using Sharplight.Graph;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// ResNet image classifiers (canonical arch <c>resnet</c>: ResNet-18 to 200 and wide ResNets as published in timm): the stem, four stages of basic or bottleneck blocks
/// (<see cref="ResNetBackbone"/>, batch norms folded into the convolutions), global average pooling and a linear classifier. Builds <see cref="WellKnown.ImageEncodeGraph"/>
/// over the processed image as one row of <c>3 · size²</c> channel-major values (the <c>patches</c> input of the image clients with a patch as large as the image) and
/// describes the model as <see cref="ModelKinds.ImageClassification"/>.
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>resnet.*</c>): <c>image_size</c>, <c>backbone.depths</c> (comma separated), <c>backbone.bottleneck</c>; the canonical keys
/// <see cref="CanonicalKeys.ClassifierLabels"/> (optional) and <see cref="CanonicalKeys.ImageProcessor"/>. Weights: <c>backbone.*</c> as for <see cref="ResNetBackbone"/>
/// and <c>classifier.{weight [classes, features], bias}</c>.
/// </remarks>
public sealed class ResNetClassifierAdapter : IModelAdapter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "resnet";

    /// <inheritdoc/>
    public string Name => "ResNet image classifier";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => [Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        int size = src.Arch<int>("image_size");
        var backbone = new ResNetOptions("backbone.", [.. src.Arch<string>("backbone.depths").Split(',').Select(int.Parse)], src.Arch("backbone.bottleneck", true), false, false);

        var b = new GraphBuilder(WellKnown.ImageEncodeGraph, src.Weight);
        var input = b.Input("patches", InputRole.Media, DataType.F32, new Shape(1, 3L * size * size));
        var stages = ResNetBackbone.Build(b, b.Reshape(input, 1, 3, size, size), backbone);
        var last = stages[^1];
        long side = size / 32;
        var pooled = b.Reshape(b.Emit1("Pool", [last], Attributes.From(("mode", "avg"), ("kernel", new long[] { side, side }), ("strides", new long[] { side, side }))), 1, last.Shape[1].Fixed);
        b.Output(WellKnown.PooledOutput, pooled);
        b.Output(WellKnown.LogitsOutput, b.Linear(pooled, "classifier.weight", "classifier.bias"));

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
}
