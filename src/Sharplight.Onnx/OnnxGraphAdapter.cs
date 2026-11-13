using Sharplight.Models;

namespace Sharplight.Onnx;

/// <summary>
/// Generic adapter that translates an ONNX graph (<see cref="OnnxModel"/> in <see cref="ModelSource.FormatData"/>)
/// op-by-op into Sharplight's IR. See <see cref="OnnxTranslator"/> for the supported operator set.
/// </summary>
public sealed class OnnxGraphAdapter : IModelAdapter
{
    /// <summary>Name of the single graph produced.</summary>
    public const string MainGraph = "main";
    /// <summary>Symbol of the batch dimension; bind it via <c>CompileOptions.SymbolBounds["N"]</c>.</summary>
    public const string BatchSymbol = OnnxTranslator.BatchSymbol;
    /// <summary>Property: batch symbol name.</summary>
    public const string BatchSymbolProperty = "onnx.batch_symbol";

    public string Name => "onnx";
    public IReadOnlyList<string> Architectures { get; } = [OnnxImporter.ArchitectureName];

    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options) => Build(sources.Primary);

    public static ModelDefinition Build(ModelSource source)
    {
        var model = source.FormatData as OnnxModel ?? throw new InvalidOperationException("ModelSource has no parsed ONNX model (FormatData).");
        var tr = new OnnxTranslator(model);
        var graph = tr.Translate(MainGraph);
        return new ModelDefinition
        {
            Architecture = OnnxImporter.ArchitectureName,
            Name = model.Graph.Name,
            Weights = tr.Weights,
            Graphs = new Dictionary<string, Graph.ComputeGraph> { [MainGraph] = graph },
            Properties = new Dictionary<string, object> { [BatchSymbolProperty] = BatchSymbol },
            SourceMetadata = source.Metadata,
            OwnedResources = source.OwnedResources,
        };
    }
}
