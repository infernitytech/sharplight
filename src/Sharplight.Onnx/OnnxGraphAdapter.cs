using Sharplight.Models;

namespace Sharplight.Onnx;

/// <summary>
/// Generic adapter that translates an ONNX graph (the parsed model carried in <see cref="ModelSource.FormatData"/>)
/// op-by-op into Sharplight's IR. See <c>OnnxTranslator</c> for the supported operator set.
/// </summary>
public sealed class OnnxGraphAdapter : IModelAdapter
{
    /// <summary>Name of the single graph produced.</summary>
    public const string MainGraph = "main";

    /// <summary>Symbol of the batch dimension; bind it via <c>CompileOptions.SymbolBounds["N"]</c>.</summary>
    public const string BatchSymbol = OnnxTranslator.BatchSymbol;

    /// <summary>Model definition property that records the batch symbol name.</summary>
    public const string BatchSymbolProperty = "onnx.batch_symbol";

    /// <inheritdoc />
    public string Name => "onnx";

    /// <inheritdoc />
    public IReadOnlyList<string> Architectures { get; } = [OnnxImporter.ArchitectureName];

    /// <inheritdoc />
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options) => Build(sources.Primary);

    /// <summary>
    /// Translates the model held by a source produced by <see cref="OnnxImporter"/> into a single-graph definition named
    /// <see cref="MainGraph"/>. The returned definition shares the source's weight mappings and takes over their ownership.
    /// </summary>
    /// <param name="source">A source created by <see cref="OnnxImporter.Open"/>.</param>
    /// <exception cref="InvalidOperationException">The source carries no parsed ONNX model.</exception>
    /// <exception cref="NotSupportedException">The graph uses an unsupported operator, dynamic dimension or data type.</exception>
    /// <exception cref="InvalidDataException">The graph is malformed (cycles, undefined values, ...).</exception>
    public static ModelDefinition Build(ModelSource source)
    {
        var model = source.FormatData as OnnxModel ?? throw new InvalidOperationException("ModelSource has no parsed ONNX model (FormatData).");
        var translator = new OnnxTranslator(model);
        var graph = translator.Translate(MainGraph);
        return new ModelDefinition
        {
            Architecture = OnnxImporter.ArchitectureName,
            Name = model.Graph.Name,
            Weights = translator.Weights,
            Graphs = new Dictionary<string, Graph.ComputeGraph> { [MainGraph] = graph },
            Properties = new Dictionary<string, object> { [BatchSymbolProperty] = BatchSymbol },
            SourceMetadata = source.Metadata,
            OwnedResources = source.OwnedResources,
        };
    }
}
