using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tokenizers;

namespace Sharplight.Tasks;

/// <summary>
/// A loaded model of any kind: its definition (built by the registered importer and architecture adapter, or read prebuilt from an
/// <c>.slm</c> package), its <see cref="ModelDescription"/> (kind and components), and the helpers task clients use to compile
/// component graphs and build component tokenizers onto the configured devices. Task clients (<c>ITextGenerator</c>,
/// <c>ISpeechRecognizer</c>, <c>ISpeechSynthesizer</c>, <c>IImageGenerator</c>, ...) are created from it by a
/// <see cref="TaskClientProvider"/>, or resolved from the service container (<c>services.AddSpeechRecognizer(...)</c>).
/// <para>
/// The model owns the weight memory: dispose it after every client created from it. Compiling and creating clients is thread-safe;
/// the clients are not unless they say so.
/// </para>
/// </summary>
/// <example>
/// <code>
/// using var services = SharplightDefaults.CreateServiceProvider();   // Sharplight.Defaults
/// using var model = TaskModel.Load(new TaskLoadOptions { ModelPath = "whisper-small.slm", Devices = [slot] });
/// using var recognizer = services.GetRequiredService&lt;TaskClientProvider&gt;().Create&lt;ISpeechRecognizer&gt;(model);
/// var transcription = await recognizer.TranscribeAsync(WavFile.Read("talk.wav"));
/// </code>
/// </example>
public sealed class TaskModel : IDisposable
{
    private const long DefaultReserve = 1L << 30;
    private readonly ConcurrentDictionary<string, Lazy<ModelDefinition>> _variants = new(StringComparer.Ordinal);
    private bool _disposed;

    internal TaskModel(ModelDefinition definition, ModelSource source, TaskLoadOptions options)
    {
        Definition = definition;
        Source = source;
        Options = options;
        Description = ModelDescription.Resolve(definition);
        Artifacts = options.Artifacts ?? new CompilationArtifacts(source.CompilationCaches);
    }

    /// <summary>The built model: architecture, weights, attachments, properties and the graphs of all components.</summary>
    public ModelDefinition Definition { get; }

    /// <summary>The canonical source the definition was built from (what a saved package stores).</summary>
    public ModelSource Source { get; }

    /// <summary>The model's kind and components (the resolved default for models that do not describe themselves).</summary>
    public ModelDescription Description { get; }

    /// <summary>The options the model was loaded with; also used for every compilation.</summary>
    public TaskLoadOptions Options { get; }

    /// <summary>The precompiled-artifact routing every compilation of this model consults and records into.</summary>
    public CompilationArtifacts Artifacts { get; }

    /// <summary>Weight buffers shared by all graphs compiled from this model, so components that share weights upload them once per device.</summary>
    public SharedWeights SharedWeights { get; } = new();

    /// <summary>
    /// Loads <see cref="TaskLoadOptions.ModelPath"/> through the registered importers and architecture adapters (the same load path as
    /// <c>LlmModel</c>). Nothing is compiled until a client or <see cref="Compile"/> asks for a graph.
    /// </summary>
    /// <param name="options">Model path, devices and compile settings; kept for every later compilation.</param>
    /// <param name="cancellationToken">Cancels loading between stages.</param>
    /// <exception cref="ArgumentException">No model path or no device is configured.</exception>
    /// <exception cref="NotSupportedException">No registered importer or architecture adapter handles the model.</exception>
    public static TaskModel Load(TaskLoadOptions options, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (string.IsNullOrEmpty(options.ModelPath))
        {
            throw new ArgumentException("TaskLoadOptions.ModelPath is not set.", nameof(options));
        }

        if (options.Devices.Count == 0)
        {
            throw new ArgumentException("TaskLoadOptions.Devices is empty: give at least one device.", nameof(options));
        }

        var stopwatch = Stopwatch.StartNew();
        var adapter = options.ContextLength > 0 && options.Adapter.ContextLength == 0 ? options.Adapter.WithContext(options.ContextLength) : options.Adapter;
        var (definition, source) = ModelLoader.Load(options.ModelPath, options.Auxiliary, adapter, options.Progress, options.Log, cancellationToken);
        var model = new TaskModel(definition, source, options);
        options.Log?.Invoke($"Loaded {definition.Name} ({definition.Architecture}, {model.Description.Kind}, components "
            + $"{string.Join(", ", model.Description.Components.Select(c => c.Role))}): {definition.Weights.Count} tensors, "
            + $"{definition.Weights.Values.Sum(w => w.ByteSize) / 1e9:F2} GB [{stopwatch.ElapsedMilliseconds} ms]");
        return model;
    }

    /// <summary>Loads <paramref name="path"/> with <paramref name="options"/> (see <see cref="Load(TaskLoadOptions, CancellationToken)"/>).</summary>
    /// <param name="path">Model file or directory; overrides <see cref="TaskLoadOptions.ModelPath"/>.</param>
    /// <param name="options">Devices and compile settings.</param>
    /// <param name="cancellationToken">Cancels loading between stages.</param>
    public static TaskModel Load(string path, TaskLoadOptions options, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(options);
        options.ModelPath = path;
        return Load(options, cancellationToken);
    }

    /// <summary>Loads the model on the thread pool (see <see cref="Load(TaskLoadOptions, CancellationToken)"/>).</summary>
    /// <param name="options">Model path, devices and compile settings.</param>
    /// <param name="cancellationToken">Cancels loading between stages.</param>
    public static Task<TaskModel> LoadAsync(TaskLoadOptions options, CancellationToken cancellationToken = default)
        => Task.Run(() => Load(options, cancellationToken), cancellationToken);

    /// <summary>
    /// The canonical metadata of a component (<see cref="ModelComponent.Metadata"/>), or of the whole model when <paramref name="role"/>
    /// is <see langword="null"/>.
    /// </summary>
    /// <param name="role">The component role, or <see langword="null"/>.</param>
    /// <exception cref="KeyNotFoundException">The model has no component with this role.</exception>
    public IReadOnlyDictionary<string, object> Metadata(string? role = null)
        => role is null ? Definition.SourceMetadata : Description.Component(role).Metadata(Definition.SourceMetadata);

    /// <summary>
    /// Builds the tokenizer of a component: its <see cref="ModelComponentAttachments.Tokenizer"/> attachment when it has one, else the
    /// tokenizer in the component's canonical metadata (or the whole model's when <paramref name="role"/> is null).
    /// </summary>
    /// <param name="role">The component role, or <see langword="null"/>.</param>
    /// <exception cref="NotSupportedException">The model describes no supported tokenizer.</exception>
    public ITokenizer CreateTokenizer(string? role = null)
    {
        if (role is not null && Description.Component(role).Attachments.ContainsKey(ModelComponentAttachments.Tokenizer))
        {
            return TokenizerFactory.Create(TokenizerData.FromJson(ReadAttachment(role, ModelComponentAttachments.Tokenizer)));
        }

        return TokenizerFactory.Create(TokenizerData.FromMetadata(Metadata(role)));
    }

    /// <summary>The bytes of an attachment of a component (<see cref="ModelComponent.Attachments"/>).</summary>
    /// <param name="role">The component role.</param>
    /// <param name="name">The attachment's local name in the component.</param>
    /// <exception cref="KeyNotFoundException">The component or attachment does not exist.</exception>
    public byte[] ReadAttachment(string role, string name)
    {
        var component = Description.Component(role);
        if (!component.Attachments.TryGetValue(name, out var stored) || !Definition.Attachments.TryGetValue(stored, out var source))
        {
            throw new KeyNotFoundException($"Component '{role}' of {Definition.Name} has no attachment '{name}'.");
        }

        var bytes = new byte[source.ByteLength];
        source.Read(0, bytes);
        return bytes;
    }

    /// <summary>A graph of a component: the one named <paramref name="graph"/>, or the component's first graph.</summary>
    /// <param name="role">The component role.</param>
    /// <param name="graph">One of the component's graph names, or <see langword="null"/> for its first graph.</param>
    /// <exception cref="KeyNotFoundException">The component does not exist or has no such graph.</exception>
    public ComputeGraph ComponentGraph(string role, string? graph = null)
    {
        var component = Description.Component(role);
        string? name = graph is null ? component.Graphs.FirstOrDefault() : component.Graphs.FirstOrDefault(g => g == graph);
        return name is not null
            ? Definition.Graph(name)
            : throw new KeyNotFoundException($"Component '{role}' of {Definition.Name} has no graph{(graph is null ? "" : $" '{graph}'")}.");
    }

    /// <summary>
    /// Compiles a graph of a component onto the configured devices, with the component's default symbol bounds
    /// (<see cref="ModelComponent.SymbolBounds"/>) under <paramref name="symbolBounds"/>; consults and records precompiled artifacts. Bounds
    /// of the row symbol (<see cref="WellKnown.TokenSymbol"/>) and the context symbol (<see cref="WellKnown.ContextSymbol"/>) replace
    /// <see cref="TaskLoadOptions.MaxBatchTokens"/> and <see cref="TaskLoadOptions.ContextLength"/> (a speech decoder's fixed context, for
    /// example), as in package precompilation.
    /// </summary>
    /// <param name="role">The component role.</param>
    /// <param name="graph">One of the component's graph names, or <see langword="null"/> for its first graph.</param>
    /// <param name="symbolBounds">Bounds that override the component's defaults, or <see langword="null"/>.</param>
    /// <exception cref="KeyNotFoundException">The component does not exist or has no such graph.</exception>
    public CompiledModel CompileComponent(string role, string? graph = null, IReadOnlyDictionary<string, long>? symbolBounds = null)
        => CompileComponent(role, graph, symbolBounds, dimensions: null);

    /// <summary>
    /// Compiles a graph of a component like <see cref="CompileComponent(string, string?, IReadOnlyDictionary{string, long}?)"/>, taken
    /// from the model rebuilt with other fixed sizes (<see cref="DefinitionFor"/>) when <paramref name="dimensions"/> is not empty.
    /// </summary>
    /// <param name="role">The component role.</param>
    /// <param name="graph">One of the component's graph names, or <see langword="null"/> for its first graph.</param>
    /// <param name="symbolBounds">Bounds that override the component's defaults, or <see langword="null"/>.</param>
    /// <param name="dimensions">Fixed sizes by name (<see cref="AdapterOptions.Dimensions"/>), or <see langword="null"/> for the built ones.</param>
    /// <exception cref="KeyNotFoundException">The component does not exist or has no such graph.</exception>
    public CompiledModel CompileComponent(string role, string? graph, IReadOnlyDictionary<string, long>? symbolBounds,
        IReadOnlyDictionary<string, long>? dimensions)
    {
        var bounds = new Dictionary<string, long>(Description.Component(role).SymbolBounds, StringComparer.Ordinal);
        foreach (var (symbol, bound) in symbolBounds ?? new Dictionary<string, long>())
        {
            bounds[symbol] = bound;
        }

        long? maxTokens = bounds.Remove(WellKnown.TokenSymbol, out long rows) ? rows : null;
        long? contextLength = bounds.Remove(WellKnown.ContextSymbol, out long context) ? context : null;
        if (dimensions is null || dimensions.Count == 0)
        {
            return Compile(ComponentGraph(role, graph), bounds, maxTokens, contextLength);
        }

        var component = Description.Component(role);
        string name = (graph is null ? component.Graphs.FirstOrDefault() : component.Graphs.FirstOrDefault(g => g == graph))
            ?? throw new KeyNotFoundException($"Component '{role}' of {Definition.Name} has no graph{(graph is null ? "" : $" '{graph}'")}.");
        return Compile(DefinitionFor(dimensions).Graph(name), bounds, maxTokens, contextLength);
    }

    /// <summary>
    /// The model rebuilt by its architecture adapter with other fixed sizes (<see cref="AdapterOptions.Dimensions"/>, e.g. the
    /// <c>height</c> and <c>width</c> of an image generator's graphs), sharing this model's weights; built once per set of sizes and
    /// kept while the model lives. Works for models loaded from any format, <c>.slm</c> packages included, because the adapter rebuilds
    /// the graphs from the canonical metadata and weights the definition keeps.
    /// </summary>
    /// <param name="dimensions">The sizes by name; empty for <see cref="Definition"/> itself.</param>
    /// <exception cref="NotSupportedException">No registered adapter builds the model's architecture.</exception>
    public ModelDefinition DefinitionFor(IReadOnlyDictionary<string, long> dimensions)
    {
        ArgumentNullException.ThrowIfNull(dimensions);
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (dimensions.Count == 0)
        {
            return Definition;
        }

        string key = string.Join(",", dimensions.OrderBy(d => d.Key, StringComparer.Ordinal)
            .Select(d => string.Create(CultureInfo.InvariantCulture, $"{d.Key}={d.Value}")));
        return _variants.GetOrAdd(key, _ => new Lazy<ModelDefinition>(() => Rebuild(dimensions))).Value;
    }

    private ModelDefinition Rebuild(IReadOnlyDictionary<string, long> dimensions)
    {
        var stopwatch = Stopwatch.StartNew();
        var source = new ModelSource
        {
            Format = Source.Format,
            Architecture = Definition.Architecture,
            Metadata = Definition.SourceMetadata,
            Weights = Definition.Weights,
            FormatData = Source.FormatData,
            Path = Source.Path,
            CompilationCaches = Source.CompilationCaches,
        };
        var definition = ModelAdapterRegistry.Build(new ModelSourceSet { Primary = source }, Options.Adapter.WithDimensions(dimensions), Options.Log);
        Options.Log?.Invoke($"Rebuilt {Definition.Name} for {string.Join(", ", dimensions.Select(d => $"{d.Key}={d.Value}"))} [{stopwatch.ElapsedMilliseconds} ms]");
        return definition;
    }

    /// <summary>
    /// Compiles one graph of the model onto the configured devices (pipeline split, fallback, stored artifacts and shared weights as
    /// configured in <see cref="Options"/>).
    /// </summary>
    /// <param name="graph">The graph, e.g. <see cref="ComponentGraph"/>.</param>
    /// <param name="symbolBounds">Upper bounds of further dynamic symbols the graph declares, or <see langword="null"/>.</param>
    /// <param name="maxTokens">Upper bound of the row symbol per run, or <see langword="null"/> for <see cref="TaskLoadOptions.MaxBatchTokens"/>.</param>
    /// <param name="contextLength">Upper bound of the context symbol, or <see langword="null"/> for <see cref="TaskLoadOptions.ContextLength"/>.</param>
    public CompiledModel Compile(ComputeGraph graph, IReadOnlyDictionary<string, long>? symbolBounds = null, long? maxTokens = null, long? contextLength = null)
    {
        ArgumentNullException.ThrowIfNull(graph);
        ObjectDisposedException.ThrowIf(_disposed, this);
        var stopwatch = Stopwatch.StartNew();
        var placement = new PlacementOptions
        {
            Devices = [.. Options.Devices],
            Fallback = Options.Fallback,
            ReservePerDevice = Options.ReservePerDevice ?? DefaultReserve,
            Compile = new CompileOptions
            {
                MaxTokens = maxTokens ?? Options.MaxBatchTokens,
                ContextLength = contextLength ?? Options.ContextLength,
                SymbolBounds = symbolBounds ?? new Dictionary<string, long>(),
                DisabledFeatures = new HashSet<string>(Options.DisabledFeatures, StringComparer.Ordinal),
                AllowReducedPrecision = Options.AllowReducedPrecision,
                CacheFor = Artifacts.For,
                Progress = Options.Progress,
                SharedWeights = SharedWeights,
            },
        };
        var compiled = CompiledModel.Compile(graph, placement);
        Options.Log?.Invoke($"Compiled '{graph.Name}' in {stopwatch.ElapsedMilliseconds} ms\n{compiled.Plan.Describe()}");
        return compiled;
    }

    /// <summary>
    /// The artifact sets every compilation of this model recorded, e.g. to store with <c>ModelPackage.AddArtifacts</c> so later loads skip
    /// kernel compilation.
    /// </summary>
    /// <param name="tiers">Only sets of these tiers (<see cref="ArtifactTiers"/>), or <see langword="null"/> for all.</param>
    public IReadOnlyList<CompilationCache> ExportCompilation(IReadOnlySet<string>? tiers = null) => Artifacts.Export(tiers);

    /// <summary>Releases the weight memory of the definition (memory maps, file handles).</summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        foreach (var resource in Definition.OwnedResources)
        {
            resource.Dispose();
        }
    }
}
