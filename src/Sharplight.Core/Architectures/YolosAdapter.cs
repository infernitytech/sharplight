using System.Text.Json;
using System.Text.Json.Nodes;
using Sharplight.Graph;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// YOLOS object detectors (canonical arch <c>yolos</c>): a plain pre-norm vision transformer over [class token, patch tokens, detection tokens] (learned position table, optional
/// per-layer "mid" position tables added after every layer but the last), then DETR's class head and three-layer box head on the detection tokens. Builds
/// <see cref="WellKnown.DetectGraph"/> for a fixed square input and describes the model as an <see cref="ModelKinds.ObjectDetection"/> model (softmax with a no-object class).
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>yolos.*</c>): <c>layer_count</c>, <c>hidden_size</c>, <c>ffn_hidden_size</c>, <c>attention.head_count</c>, <c>norm_eps</c>, <c>mlp.activation</c>,
/// <c>patch_size</c>, <c>image_size</c> (square input side), <c>detection_token_count</c>, <c>mid_position_embeddings</c>; canonical <see cref="CanonicalKeys.ClassifierLabels"/> and
/// <see cref="CanonicalKeys.ImageProcessor"/>. Weights: <c>patch_embed.{weight [d, 3, p, p],bias}</c>, <c>class_embedding</c> [d], <c>detection_embedding.weight</c> [T, d],
/// <c>position_embedding.weight</c> [1 + grid² + T, d] (already at the input's grid), optional <c>mid_position_embedding.weight</c> [layers - 1, 1 + grid² + T, d], the layers of
/// <see cref="TowerBlocks.Layers"/> (<c>layers.{i}.</c>), <c>output_norm</c>, <c>class_head.{0,1,2}</c> and <c>box_head.{0,1,2}</c>.
/// </remarks>
public sealed class YolosAdapter : IModelAdapter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "yolos";

    /// <inheritdoc/>
    public string Name => "YOLOS object detector";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => [Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        int hidden = src.Arch<int>("hidden_size");
        int patch = src.Arch<int>("patch_size");
        int size = src.Arch<int>("image_size");
        int detection = src.Arch<int>("detection_token_count");
        int grid = size / patch;
        var tower = new TowerOptions(src.Arch<int>("layer_count"), hidden, src.Arch<int>("attention.head_count"), src.Arch("norm_eps", 1e-12f), src.Arch("mlp.activation", "gelu"), "");
        bool mid = src.Arch("mid_position_embeddings", false);
        var graph = BuildGraph();
        var labels = src.Get(CanonicalKeys.ClassifierLabels, Array.Empty<string>());
        var processor = JsonNode.Parse(src.Get<string>(CanonicalKeys.ImageProcessor))!;
        var description = new ModelDescription
        {
            Kind = ModelKinds.ObjectDetection,
            Config = ComponentConfig.Of(("scoring", "softmax-background")).Concat(
                new Dictionary<string, JsonElement> { ["labels"] = JsonSerializer.SerializeToElement(labels) }).ToDictionary(),
            Components =
            [
                new ModelComponent
                {
                    Role = ModelComponentRoles.FeatureExtractor,
                    Config = new Dictionary<string, JsonElement> { [ImageProcessor.ComponentKey] = JsonSerializer.SerializeToElement(processor) },
                },
                new ModelComponent { Role = ModelComponentRoles.VisionEncoder, Graphs = [WellKnown.DetectGraph], Config = ComponentConfig.Of(("imageSize", size)) },
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

        ComputeGraph BuildGraph()
        {
            var b = new GraphBuilder(WellKnown.DetectGraph, src.Weight);
            var pixels = b.Input("pixels", InputRole.Media, DataType.F32, new Shape(1, 3, size, size));
            int rows = 1 + grid * grid + detection;
            Value x;
            using (b.Group("embedding"))
            {
                var projected = b.Emit1("Conv", [pixels, b.Weight("patch_embed.weight"), b.Weight("patch_embed.bias")], Attributes.From(("strides", new long[] { patch, patch }), ("pads", new long[] { 0, 0, 0, 0 })));
                var patches = b.Contiguous(b.Permute(b.Reshape(projected, hidden, grid * grid), 1, 0));
                x = b.Concat(0, b.Reshape(b.Weight("class_embedding"), 1, hidden), patches, b.Weight("detection_embedding.weight"));
                x = b.Add(x, b.Weight("position_embedding.weight"));
            }

            for (int il = 0; il < tower.Layers; il++)
            {
                x = TowerBlocks.Layers(b, x, tower with { Layers = 1 }, rows, causal: false, layerOffset: il);
                if (mid && il < tower.Layers - 1)
                {
                    x = b.Add(x, b.Reshape(b.Slice(b.Weight("mid_position_embedding.weight"), 0, il, 1), rows, hidden));
                }
            }

            using (b.Group("heads"))
            {
                x = TowerBlocks.LayerNorm(b, x, "output_norm", tower.Eps);
                var queries = b.Slice(x, 0, rows - detection, detection);
                var logits = queries;
                var box = queries;
                for (int i = 0; i < 3; i++)
                {
                    logits = b.Linear(logits, $"class_head.{i}.weight", $"class_head.{i}.bias");
                    box = b.Linear(box, $"box_head.{i}.weight", $"box_head.{i}.bias");
                    if (i < 2)
                    {
                        logits = b.Unary("Relu", logits);
                        box = b.Unary("Relu", box);
                    }
                }

                b.Output(WellKnown.LogitsOutput, logits);
                b.Output(WellKnown.BoxesOutput, b.Sigmoid(box));
            }

            return b.Build();
        }
    }
}
