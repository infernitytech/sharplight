using Sharplight.Graph.Optimization;
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

    /// <summary>
    /// The name of the graph input or output that carries element <paramref name="index"/> of an ONNX sequence value: a sequence graph
    /// input becomes one input per element (its length is given by <see cref="OnnxImportOptions.SequenceInputLengths"/>, the element
    /// shapes by <see cref="OnnxImportOptions.InputShapes"/> under these names), and a sequence graph output whose length is known at
    /// import one output per element. A sequence output whose length is only known at run time is one output under the sequence's own
    /// name holding the elements stacked along a new outermost axis. An empty optional output has no graph output.
    /// </summary>
    /// <param name="sequence">The ONNX name of the sequence value.</param>
    /// <param name="index">The element index.</param>
    /// <returns><c>sequence[index]</c>.</returns>
    public static string SequenceElementName(string sequence, int index) => $"{sequence}[{index}]";

    /// <inheritdoc />
    public string Name => "onnx";

    /// <inheritdoc />
    public IReadOnlyList<string> Architectures { get; } = [OnnxImporter.ArchitectureName];

    /// <inheritdoc />
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options) => Build(sources.Primary, options.OptimizeGraphs);

    /// <summary>
    /// Translates the model held by a source produced by <see cref="OnnxImporter"/> into a single-graph definition named
    /// <see cref="MainGraph"/>, then runs the graph optimization phase (<see cref="GraphOptimizer.PrepareForLoad"/>) unless
    /// <paramref name="optimizeGraphs"/> is false. The returned definition shares the source's weight mappings and takes over their ownership.
    /// </summary>
    /// <param name="source">A source created by <see cref="OnnxImporter.Open"/>.</param>
    /// <param name="optimizeGraphs">Whether to run the graph optimization phase (it is also skipped when <c>SHARPLIGHT_NO_GRAPH_OPT=1</c>).</param>
    /// <exception cref="InvalidOperationException">The source carries no parsed ONNX model.</exception>
    /// <exception cref="NotSupportedException">The graph uses an unsupported operator, dynamic dimension or data type.</exception>
    /// <exception cref="InvalidDataException">The graph is malformed (cycles, undefined values, ...).</exception>
    public static ModelDefinition Build(ModelSource source, bool optimizeGraphs = true)
    {
        var model = source.FormatData as OnnxModel ?? throw new InvalidOperationException("ModelSource has no parsed ONNX model (FormatData).");
        var translator = new OnnxTranslator(model);
        var graph = translator.Translate(MainGraph);
        var definition = new ModelDefinition
        {
            Architecture = OnnxImporter.ArchitectureName,
            Name = model.Graph.Name,
            Weights = translator.Weights,
            Graphs = new Dictionary<string, Graph.ComputeGraph> { [MainGraph] = graph },
            Properties = new Dictionary<string, object> { [BatchSymbolProperty] = BatchSymbol },
            SourceMetadata = source.Metadata,
            OwnedResources = source.OwnedResources,
        };
        return GraphOptimizer.PrepareForLoad(definition, optimizeGraphs);
    }
}
