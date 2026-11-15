using System.Diagnostics;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tokenizers;

namespace Sharplight.Tasks;

/// <summary>
/// A loaded model for the generic task layer: its definition (built by the registered importer and architecture adapter, or read
/// prebuilt from an <c>.slm</c> package), its <see cref="ModelTaskDescriptor"/>, and helpers to compile component graphs and build
/// component tokenizers onto the configured devices. Task utilities (<see cref="Speech.ISpeechRecognizer"/>,
/// <see cref="ImageGeneration.IImageGenerator"/>, ...) are created from it through <see cref="ModelTaskRegistry"/>, usually with
/// the extension methods of their namespace (<c>model.CreateSpeechRecognizer()</c>).
/// <para>
/// The model owns the weight memory: dispose it after everything created from it. Compiling and creating task utilities is
/// thread-safe; the created objects are not unless they say so.
/// </para>
/// </summary>
/// <example>
/// <code>
/// SharplightDefaults.Register();
/// using var model = TaskModel.Load("whisper-small.slm", new TaskLoadOptions { Devices = [slot] });
/// using var recognizer = model.CreateSpeechRecognizer();
/// var transcription = recognizer.Transcribe(WavFile.Read("talk.wav"));
/// </code>
/// </example>
public sealed class TaskModel : IDisposable
{
    private const long DefaultReserve = 1L << 30;

    private TaskModel(ModelDefinition definition, ModelSource source, TaskLoadOptions options)
    {
        Definition = definition;
        Source = source;
        Options = options;
        Task = ModelTaskDescriptor.Read(definition.Properties);
        Artifacts = options.Artifacts ?? new CompilationArtifacts(source.CompilationCaches);
    }

    /// <summary>The built model: architecture, weights, properties and the graphs of all components.</summary>
    public ModelDefinition Definition { get; }

    /// <summary>The canonical source the definition was built from (what a saved package stores).</summary>
    public ModelSource Source { get; }

    /// <summary>The task descriptor, or <see langword="null"/> when the model declares none.</summary>
    public ModelTaskDescriptor? Task { get; }

    /// <summary>The options the model was loaded with; also used for every compilation.</summary>
    public TaskLoadOptions Options { get; }

    /// <summary>The precompiled-artifact routing every compilation of this model consults and records into.</summary>
    public CompilationArtifacts Artifacts { get; }

    /// <summary>Weight buffers shared by all graphs compiled from this model, so components that share weights upload them once per device.</summary>
    public SharedWeights SharedWeights { get; } = new();

    /// <summary>
    /// Loads a model through <see cref="ModelImporterRegistry"/> and <see cref="ModelAdapterRegistry"/>; the host must have registered
    /// the importers and adapters (<c>SharplightDefaults.Register()</c> of the Sharplight.Defaults package does both). Nothing is
    /// compiled until a task utility or <see cref="Compile"/> asks for a graph.
    /// </summary>
    /// <param name="path">Model file or directory, as understood by the registered importers.</param>
    /// <param name="options">Devices and compile settings.</param>
    /// <exception cref="NotSupportedException">No registered importer or architecture adapter handles the model.</exception>
    public static TaskModel Load(string path, TaskLoadOptions options)
    {
        var stopwatch = Stopwatch.StartNew();
        options.CancellationToken.ThrowIfCancellationRequested();
        var reading = new LoadProgressCounter(options.Progress, LoadStage.Reading, 1, path);
        var sources = ModelImporterRegistry.OpenSet(path, [.. options.Auxiliary]);
        reading.Complete();
        options.CancellationToken.ThrowIfCancellationRequested();
        var building = new LoadProgressCounter(options.Progress, LoadStage.Building, 1);
        var definition = ModelAdapterRegistry.Build(sources, options.Adapter, options.Log);
        building.Complete();
        var model = new TaskModel(definition, sources.Primary, options);
        options.Log?.Invoke($"Loaded {definition.Name} ({definition.Architecture}, task {model.Task?.Kind ?? "unspecified"}): "
            + $"{definition.Weights.Count} tensors, {definition.Weights.Values.Sum(w => w.ByteSize) / 1e9:F2} GB [{stopwatch.ElapsedMilliseconds} ms]");
        return model;
    }

    /// <summary>The task descriptor.</summary>
    /// <exception cref="InvalidOperationException">The model declares no task descriptor.</exception>
    public ModelTaskDescriptor RequireTask()
        => Task ?? throw new InvalidOperationException($"{Definition.Name} ({Definition.Architecture}) declares no task descriptor.");

    /// <summary>
    /// The canonical metadata of a component (<see cref="TaskComponent.Metadata"/>), or of the whole model when <paramref name="role"/>
    /// is <see langword="null"/>.
    /// </summary>
    /// <exception cref="InvalidOperationException">A role was given but the model declares no task descriptor.</exception>
    /// <exception cref="KeyNotFoundException">The model has no component with this role.</exception>
    public IReadOnlyDictionary<string, object> Metadata(string? role = null)
        => role is null ? Definition.SourceMetadata : RequireTask().Component(role).Metadata(Definition.SourceMetadata);

    /// <summary>Builds the tokenizer stored in the canonical metadata of a component, or of the whole model when <paramref name="role"/> is null.</summary>
    /// <exception cref="NotSupportedException">The metadata describes no supported tokenizer.</exception>
    public ITokenizer CreateTokenizer(string? role = null) => TokenizerFactory.Create(TokenizerData.FromMetadata(Metadata(role)));

    /// <summary>The graph of a component.</summary>
    /// <exception cref="KeyNotFoundException">The component does not exist or names no graph of the model.</exception>
    public ComputeGraph ComponentGraph(string role)
    {
        var component = RequireTask().Component(role);
        return component.Graph is { } graph
            ? Definition.Graph(graph)
            : throw new KeyNotFoundException($"Component '{role}' of {Definition.Name} has no graph.");
    }

    /// <summary>
    /// Compiles one graph of the model onto the configured devices (pipeline split, fallback, stored artifacts and shared weights as
    /// configured in <see cref="Options"/>).
    /// </summary>
    /// <param name="graph">The graph, e.g. <see cref="ComponentGraph"/>.</param>
    /// <param name="maxTokens">Upper bound of the row symbol (tokens, frames, latent positions) per run.</param>
    /// <param name="contextLength">Upper bound of the context symbol (cached positions) for graphs with persistent state.</param>
    /// <param name="symbolBounds">Upper bounds of further dynamic symbols the graph declares, or <see langword="null"/>.</param>
    public CompiledModel Compile(ComputeGraph graph, long maxTokens, long contextLength, IReadOnlyDictionary<string, long>? symbolBounds = null)
    {
        var stopwatch = Stopwatch.StartNew();
        var placement = new PlacementOptions
        {
            Devices = Options.Devices,
            Fallback = Options.Fallback,
            ReservePerDevice = Options.ReservePerDevice ?? DefaultReserve,
            Compile = new CompileOptions
            {
                MaxTokens = maxTokens,
                ContextLength = contextLength,
                SymbolBounds = symbolBounds ?? new Dictionary<string, long>(),
                DisabledFeatures = Options.DisabledFeatures,
                AllowReducedPrecision = Options.AllowReducedPrecision,
                CacheFor = Artifacts.For,
                Progress = Options.Progress,
                CancellationToken = Options.CancellationToken,
                SharedWeights = SharedWeights,
            },
        };
        var compiled = CompiledModel.Compile(graph, placement);
        Options.Log?.Invoke($"Compiled '{graph.Name}' in {stopwatch.ElapsedMilliseconds} ms\n{compiled.Plan.Describe()}");
        return compiled;
    }

    /// <summary>Releases the weight memory of the definition (memory maps, file handles).</summary>
    public void Dispose()
    {
        foreach (var resource in Definition.OwnedResources)
        {
            resource.Dispose();
        }
    }
}
