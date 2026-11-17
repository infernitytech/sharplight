using System.Text.Json;
using System.Text.Json.Nodes;
using Sharplight.Graph;
using Sharplight.Imaging;
using Sharplight.Import.Onnx;
using Sharplight.Models;
using Sharplight.Vision;

namespace Sharplight.Import.Ultralytics;

/// <summary>
/// Builds the object detection model of an Ultralytics export: the ONNX graph as the <see cref="WellKnown.DetectGraph"/> of a vision encoder component, the letterbox preprocessing
/// (longest side scaled into the input size, gray 114 padding, 0..1 range) and the head layout the dense detector client decodes (see <c>denseHead</c> in the component configuration).
/// </summary>
public sealed class UltralyticsAdapter : IModelAdapter
{
    /// <inheritdoc/>
    public string Name => "Ultralytics YOLO";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => [UltralyticsImporter.Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        var onnx = OnnxGraphAdapter.Build(src, options.OptimizeGraphs);
        if (onnx.Properties.ContainsKey(OnnxGraphAdapter.OutputLayoutsProperty))
        {
            throw new NotSupportedException("The export has an output whose layout differs from the ONNX tensor; not a YOLO head.");
        }

        var graph = onnx.Graphs[OnnxGraphAdapter.MainGraph];
        var input = graph.Inputs[0];
        if (input.Shape.Rank != 4 || input.Shape[1].Fixed != 3)
        {
            throw new NotSupportedException($"The export's image input must be [N, 3, H, W] with fixed sizes, not {input.Shape}.");
        }

        int height = (int)input.Shape[2].Fixed;
        int width = (int)input.Shape[3].Fixed;
        var outputs = graph.Outputs.ToList();
        var main = outputs[0];
        bool endToEnd = main.Value.Shape.Rank == 3 && main.Value.Shape[2].Fixed == 6;
        string? protos = outputs.Count > 1 && outputs[1].Value.Shape.Rank == 4 ? outputs[1].Key : null;
        var labels = src.Get(CanonicalKeys.ClassifierLabels, Array.Empty<string>());
        long channels = main.Value.Shape[1].Fixed;
        if (labels.Length == 0 && !endToEnd)
        {
            long masks = protos is null ? 0 : outputs[1].Value.Shape[1].Fixed;
            labels = channels - 4 - masks == CocoLabels.Names.Length ? CocoLabels.Names : [];
        }

        var processor = new ImageProcessorConfig
        {
            ResizeMode = ImageResizeMode.Contain,
            Width = width,
            Height = height,
            CropWidth = width,
            CropHeight = height,
            Filter = ImageResampleFilter.Bilinear,
            PadColor = 114,
            Rescale = 1f / 255f,
        };
        var head = new JsonObject { ["layout"] = endToEnd ? "end-to-end" : "channels-first", ["output"] = main.Key };
        if (protos is not null)
        {
            head["protos"] = protos;
        }

        var description = new ModelDescription
        {
            Kind = ModelKinds.ObjectDetection,
            Config = new Dictionary<string, JsonElement> { ["labels"] = JsonSerializer.SerializeToElement(labels) },
            Components =
            [
                new ModelComponent
                {
                    Role = ModelComponentRoles.FeatureExtractor,
                    Config = new Dictionary<string, JsonElement> { [ImageProcessor.ComponentKey] = JsonSerializer.SerializeToElement(processor.ToJson()) },
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.VisionEncoder,
                    Graphs = [WellKnown.DetectGraph],
                    SymbolBounds = new Dictionary<string, long> { [OnnxGraphAdapter.BatchSymbol] = 1 },
                    Config = new Dictionary<string, JsonElement>
                    {
                        ["imageSize"] = JsonSerializer.SerializeToElement(Math.Max(width, height)),
                        ["denseHead"] = JsonSerializer.SerializeToElement(head),
                    },
                },
            ],
        };
        return new ModelDefinition
        {
            Architecture = src.Architecture,
            Name = src.Get(CanonicalKeys.Name, src.Architecture),
            Weights = onnx.Weights,
            Graphs = new Dictionary<string, ComputeGraph> { [WellKnown.DetectGraph] = graph },
            Properties = onnx.Properties,
            SourceMetadata = src.Metadata,
            OwnedResources = onnx.OwnedResources,
            Description = description,
        };
    }
}
