using System.Text.Json;
using System.Text.Json.Nodes;
using Sharplight.Graph;
using Sharplight.Imaging;
using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// Plain vision transformer image encoders (canonical arch <c>vit</c>): ViT / DeiT classifiers, DINOv2 (with LayerScale and register
/// tokens) and the like. Builds <see cref="WellKnown.ImageEncodeGraph"/> over the host-patchified image and describes the model as an
/// <see cref="ModelKinds.ImageClassification"/> (classifier head present) or <see cref="ModelKinds.ImageEmbedding"/> model with a
/// <see cref="ModelComponentRoles.FeatureExtractor"/> (the stored <see cref="ImageProcessorConfig"/>) and a
/// <see cref="ModelComponentRoles.VisionEncoder"/> component; the Core image clients run it.
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>vit.*</c>): <c>layer_count</c>, <c>hidden_size</c>, <c>ffn_hidden_size</c>, <c>attention.head_count</c>, <c>norm_eps</c>,
/// <c>mlp.activation</c> ("gelu" (erf, default), "gelu_tanh", "quick_gelu", "relu", "silu"), <c>patch_size</c>, <c>image_size</c> (the side of
/// the square input; the position table has exactly the grid of this size), <c>register_count</c> (default 0), <c>pooling</c> ("cls" (default)
/// or "mean": the mean of the patch rows), <c>rope_theta</c> (DINOv3: rotary positions on the patch rows instead of a position table, default 0). The canonical keys <see cref="CanonicalKeys.ClassifierLabels"/>, <see cref="CanonicalKeys.ClassifierActivation"/>
/// and <see cref="CanonicalKeys.ImageProcessor"/> describe the model's classes and preprocessing.
/// <para>Weights: <c>patch_embed.weight</c> [hidden, 3·patch²] (the patch convolution flattened channel-major) and <c>.bias</c>;
/// <c>class_embedding</c> [hidden]; optional <c>register_embedding.weight</c> [registers, hidden]; <c>position_embedding.weight</c>
/// [1 + grid², hidden] (row 0 is the class token's); per layer <c>layers.{i}.</c>: <c>input_norm</c>, <c>attention.{q,k,v,o}</c> (+ biases),
/// optional <c>layer_scale_1.weight</c> [hidden], <c>post_attention_norm</c>, <c>mlp.{up,down}</c> (+ biases, and <c>mlp.gate</c> for gated MLPs), optional <c>layer_scale_2.weight</c>;
/// <c>output_norm</c> (applied to every row before pooling); optional <c>classifier.out.{weight,bias}</c> [labels, hidden]. Norms are LayerNorms.</para>
/// </remarks>
public sealed class ImageEncoderAdapter : IModelAdapter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "vit";

    /// <inheritdoc/>
    public string Name => "Vision transformer image encoder";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => [Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        int layers = src.Arch<int>("layer_count");
        int hidden = src.Arch<int>("hidden_size");
        int heads = src.Arch<int>("attention.head_count");
        int patch = src.Arch<int>("patch_size");
        int imageSize = src.Arch<int>("image_size");
        int registers = src.Arch("register_count", 0);
        float eps = src.Arch("norm_eps", 1e-6f);
        string act = src.Arch("mlp.activation", "gelu");
        string pooling = src.Arch("pooling", "cls");
        bool classifier = src.Weight("classifier.out.weight") is not null;
        var graph = BuildGraph();

        var config = new Dictionary<string, JsonElement>();
        if (classifier)
        {
            var labels = src.Get(CanonicalKeys.ClassifierLabels, Array.Empty<string>());
            config["labels"] = JsonSerializer.SerializeToElement(labels);
            config["scoring"] = JsonSerializer.SerializeToElement(src.Get(CanonicalKeys.ClassifierActivation, "softmax"));
        }

        var processor = JsonNode.Parse(src.Get<string>(CanonicalKeys.ImageProcessor))!;
        var description = new ModelDescription
        {
            Kind = classifier ? ModelKinds.ImageClassification : ModelKinds.ImageEmbedding,
            Config = config,
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
                        ["patchSize"] = JsonSerializer.SerializeToElement(patch),
                        ["imageSize"] = JsonSerializer.SerializeToElement(imageSize),
                    },
                },
            ],
        };
        return new ModelDefinition
        {
            Architecture = src.Architecture,
            Name = src.Get(CanonicalKeys.Name, src.Architecture),
            Weights = src.Weights,
            Graphs = new Dictionary<string, ComputeGraph> { [WellKnown.ImageEncodeGraph] = graph },
            SourceMetadata = src.Metadata,
            OwnedResources = src.OwnedResources,
            Description = description,
        };

        ComputeGraph BuildGraph() => ImageTowerGraph.Build(WellKnown.ImageEncodeGraph, src, new ImageTowerOptions(
            new TowerOptions(layers, hidden, heads, eps, act, ""), patch, imageSize, registers, pooling, src.Arch("rope_theta", 0f)));
    }
}
