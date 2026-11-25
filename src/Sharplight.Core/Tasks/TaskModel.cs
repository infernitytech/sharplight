using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.Text;
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
/// using var services = SharplightDefaults.CreateServiceProvider();   // Sharplight
/// var loader = services.GetRequiredService&lt;ModelLoader&gt;();
/// using var model = TaskModel.Load(loader, new TaskLoadOptions { ModelPath = "whisper-small.slm", Devices = [slot] });
/// using var recognizer = services.GetRequiredService&lt;TaskClientProvider&gt;().Create&lt;ISpeechRecognizer&gt;(model);
/// var transcription = await recognizer.TranscribeAsync(WavFile.Read("talk.wav"));
/// </code>
/// </example>
public sealed class TaskModel : IDisposable
{
    private const long DefaultReserve = 1L << 30;
    private const string ActivationRangeKeyPrefix = "activation-range|";
    private readonly ConcurrentDictionary<string, Lazy<ModelDefinition>> _variants = new(StringComparer.Ordinal);
    private readonly ConcurrentDictionary<string, Lazy<ActivationPrecision>> _activationPrecisions = new(StringComparer.Ordinal);
    private bool _disposed;

    internal TaskModel(ModelDefinition definition, ModelSource source, TaskLoadOptions options, ArchitectureCatalog architectures)
    {
        Architectures = architectures;
        Definition = definition;
        Source = source;
        Options = options;
        Description = ModelDescription.Resolve(definition);
        Artifacts = options.Artifacts ?? new CompilationArtifacts(source.CompilationCaches);
        _diskExperts = new Lazy<Runtime.Experts.ExpertStore?>(() => Runtime.Experts.ExpertStore.ForLoad(options));
    }

    /// <summary>The built model: architecture, weights, attachments, properties and the graphs of all components.</summary>
    public ModelDefinition Definition { get; }

    /// <summary>The canonical source the definition was built from (what a saved package stores).</summary>
    public ModelSource Source { get; }

    /// <summary>The model's kind and components (the resolved default for models that do not describe themselves).</summary>
    public ModelDescription Description { get; }

    /// <summary>The architectures the model was built with (for task clients that need architecture services such as prompt assembly).</summary>
    public ArchitectureCatalog Architectures { get; }

    /// <summary>The options the model was loaded with; also used for every compilation.</summary>
    public TaskLoadOptions Options { get; }

    /// <summary>The precompiled-artifact routing every compilation of this model consults and records into.</summary>
    public CompilationArtifacts Artifacts { get; }

    /// <summary>Weight buffers shared by all graphs compiled from this model, so components that share weights upload them once per device.</summary>
    public SharedWeights SharedWeights { get; } = new();

    /// <summary>
    /// The routed experts left on disk (<see cref="TaskLoadOptions.ExpertsOnDisk"/>), shared by all graphs compiled from this model, or null when
    /// the option is off. <see cref="Runtime.Experts.ExpertStore.Statistics"/> has the cache's hit rate and the bytes read.
    /// </summary>
    public Runtime.Experts.ExpertStore? DiskExperts => _diskExperts.Value;

    private readonly Lazy<Runtime.Experts.ExpertStore?> _diskExperts;

    /// <summary>
    /// Loads <see cref="TaskLoadOptions.ModelPath"/> through the loader's importers and architectures (the same load path as
    /// <c>LlmModel</c>). Nothing is compiled until a client or <see cref="Compile"/> asks for a graph.
    /// </summary>
    /// <param name="loader">The model loader (from the service container: <c>services.AddSharplight()</c>).</param>
    /// <param name="options">Model path, devices and compile settings; kept for every later compilation.</param>
    /// <param name="cancellationToken">Cancels loading between stages.</param>
    /// <returns>The loaded model; the caller disposes it after every client created from it.</returns>
    /// <exception cref="ArgumentException">No model path or no device is configured.</exception>
    /// <exception cref="NotSupportedException">No registered importer or architecture adapter handles the model.</exception>
    public static TaskModel Load(ModelLoader loader, TaskLoadOptions options, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(loader);
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
        var (definition, source) = loader.Load(options.ModelPath, options.Auxiliary, adapter, options.Progress, options.Log, cancellationToken);
        var model = new TaskModel(definition, source, options, loader.Architectures);
        options.Log?.Invoke($"Loaded {definition.Name} ({definition.Architecture}, {model.Description.Kind}, components "
            + $"{string.Join(", ", model.Description.Components.Select(c => c.Role))}): {definition.Weights.Count} tensors, "
            + $"{definition.Weights.Values.Sum(w => w.ByteSize) / 1e9:F2} GB [{stopwatch.ElapsedMilliseconds} ms]");
        return model;
    }

    /// <summary>Loads the model on a background thread, returning at once (see <see cref="Load(ModelLoader, TaskLoadOptions, CancellationToken)"/>).</summary>
    /// <param name="loader">The model loader.</param>
    /// <param name="options">Model path, devices and compile settings.</param>
    /// <param name="cancellationToken">Cancels loading between stages.</param>
    /// <returns>The loaded model.</returns>
    public static Task<TaskModel> LoadAsync(ModelLoader loader, TaskLoadOptions options, CancellationToken cancellationToken = default)
        => BackgroundCompute.RunAsync(() => Load(loader, options, cancellationToken), cancellationToken);

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
        => CompileComponent(role, ResolveComponentGraph(role, graph, dimensions), symbolBounds);

    /// <summary>
    /// The activation precision (<see cref="CompileOptions.ActivationPrecision"/>) every later compilation of a component graph uses,
    /// decided once per graph name: <see cref="ActivationPrecision.BFloat16"/> when the graph's activations exceed f16's safe range
    /// (<see cref="ActivationRange.FitsHalf"/>), measured by one run of the graph instrumented with <see cref="ActivationRangeProbe"/>,
    /// otherwise <see cref="ActivationPrecision.Half"/>. Nothing is measured, and the result is <see cref="ActivationPrecision.Half"/>,
    /// when <see cref="TaskLoadOptions.AllowReducedPrecision"/> is off or no configured device multiplies f16 matrices (the CPU). The
    /// measured range is recorded in the model's compilation artifacts (<see cref="Artifacts"/>, stored with a package by
    /// <see cref="ExportCompilation"/>) and read from there by later loads. Thread-safe; concurrent callers for one graph share one
    /// measurement.
    /// </summary>
    /// <param name="role">The component role.</param>
    /// <param name="graph">One of the component's graph names.</param>
    /// <param name="dimensions">Fixed sizes to measure at (<see cref="DefinitionFor"/>), small ones being cheapest; <see langword="null"/>
    /// for the built ones.</param>
    /// <param name="prepare">Sets the inputs and symbols of the measuring run: a small, representative input.</param>
    /// <returns>The precision the graph's compilations use from now on.</returns>
    /// <exception cref="KeyNotFoundException">The component does not exist or has no such graph.</exception>
    public ActivationPrecision CalibrateActivationPrecision(string role, string graph, IReadOnlyDictionary<string, long>? dimensions,
        Action<CompiledModel, RunParameters> prepare)
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(prepare);
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!Options.AllowReducedPrecision || !Options.Devices.Any(slot => HasHalfMatrices(slot.Device)))
        {
            return ActivationPrecision.Half;
        }

        return _activationPrecisions.GetOrAdd(graph, _ => new Lazy<ActivationPrecision>(() => MeasureActivationPrecision(role, graph, dimensions, prepare))).Value;
    }

    private ActivationPrecision MeasureActivationPrecision(string role, string graph, IReadOnlyDictionary<string, long>? dimensions,
        Action<CompiledModel, RunParameters> prepare)
    {
        var stopwatch = Stopwatch.StartNew();
        var instrumented = ActivationRangeProbe.Instrument(ResolveComponentGraph(role, graph, dimensions));
        string key = ActivationRangeKeyPrefix + Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(Encoding.UTF8.GetBytes(instrumented.Dump())), 0, 12);
        var cache = Options.Devices.Select(slot => Artifacts.For(slot.Device)).FirstOrDefault(c => c is not null);
        ActivationRange range;
        if (cache is not null && cache.TryGet(key, out var stored) && stored.Length == sizeof(float))
        {
            range = new ActivationRange(BitConverter.ToSingle(stored.Span), Location: null);
        }
        else
        {
            try
            {
                // Strict products: every activation exact, none lost to an earlier overflow.
                using var compiled = CompileComponent(role, instrumented, symbolBounds: null, allowReducedPrecision: false);
                var parameters = new RunParameters();
                prepare(compiled, parameters);
                compiled.Run(parameters);
                range = ActivationRangeProbe.Read(compiled, parameters);
            }
            catch (InsufficientMemoryException e)
            {
                // Not measurable now (the device is full): keep f16 as without the measurement, and measure again next load.
                Options.Log?.Invoke($"Activation range of '{graph}' not measured: {e.Message}");
                return ActivationPrecision.Half;
            }

            cache?.Put(key, BitConverter.GetBytes(range.MaxMagnitude));
        }

        Options.Log?.Invoke(string.Create(CultureInfo.InvariantCulture,
            $"Activation range of '{graph}': max |x| {range.MaxMagnitude:G4}{(range.Location is null ? " (stored)" : $" at '{range.Location}'")} -> {range.Precision} activations [{stopwatch.ElapsedMilliseconds} ms]"));
        return range.Precision;
    }

    private bool HasHalfMatrices(IComputeDevice device)
    {
        var features = DeviceFeatures.Effective(device.EnabledFeatures, Options.CompileDisabledFeatures());
        return features.Contains(DeviceFeatures.MatrixF16) || features.Contains(DeviceFeatures.MatrixF16Small);
    }

    private ComputeGraph ResolveComponentGraph(string role, string? graph, IReadOnlyDictionary<string, long>? dimensions)
    {
        if (dimensions is null || dimensions.Count == 0)
        {
            return ComponentGraph(role, graph);
        }

        var component = Description.Component(role);
        string name = (graph is null ? component.Graphs.FirstOrDefault() : component.Graphs.FirstOrDefault(g => g == graph))
            ?? throw new KeyNotFoundException($"Component '{role}' of {Definition.Name} has no graph{(graph is null ? "" : $" '{graph}'")}.");
        return DefinitionFor(dimensions).Graph(name);
    }

    private CompiledModel CompileComponent(string role, ComputeGraph graph, IReadOnlyDictionary<string, long>? symbolBounds, bool? allowReducedPrecision = null)
    {
        var bounds = new Dictionary<string, long>(Description.Component(role).SymbolBounds, StringComparer.Ordinal);
        foreach (var (symbol, bound) in symbolBounds ?? new Dictionary<string, long>())
        {
            bounds[symbol] = bound;
        }

        long? maxTokens = bounds.Remove(WellKnown.TokenSymbol, out long rows) ? rows : null;
        long? contextLength = bounds.Remove(WellKnown.ContextSymbol, out long context) ? context : null;
        return Compile(graph, bounds, maxTokens, contextLength, allowReducedPrecision ?? Options.AllowReducedPrecision);
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
        var definition = Architectures.Build(new ModelSourceSet { Primary = source }, Options.Adapter.WithDimensions(dimensions), Options.Log);
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
        => Compile(graph, symbolBounds, maxTokens, contextLength, Options.AllowReducedPrecision);

    private CompiledModel Compile(ComputeGraph graph, IReadOnlyDictionary<string, long>? symbolBounds, long? maxTokens, long? contextLength,
        bool allowReducedPrecision)
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
                DisabledFeatures = Options.CompileDisabledFeatures(),
                AllowReducedPrecision = allowReducedPrecision,
                ActivationPrecision = ActivationPrecisionOf(graph.Name),
                CacheFor = Artifacts.For,
                Progress = Options.Progress,
                SharedWeights = SharedWeights,
                DiskExperts = DiskExperts,
            },
        };
        var compiled = CompiledModel.Compile(graph, placement);
        Options.Log?.Invoke($"Compiled '{graph.Name}' in {stopwatch.ElapsedMilliseconds} ms\n{compiled.Plan.Describe()}");
        return compiled;
    }

    /// <summary>
    /// The activation precision compilations of the graph named <paramref name="graph"/> use: what
    /// <see cref="CalibrateActivationPrecision"/> decided for it, else <see cref="ActivationPrecision.Half"/>.
    /// </summary>
    /// <param name="graph">A graph name.</param>
    public ActivationPrecision ActivationPrecisionOf(string graph)
        => _activationPrecisions.TryGetValue(graph, out var decided) && decided.IsValueCreated ? decided.Value : ActivationPrecision.Half;

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
        if (_diskExperts.IsValueCreated)
        {
            _diskExperts.Value?.Dispose();
        }

        foreach (var resource in Definition.OwnedResources)
        {
            resource.Dispose();
        }
    }
}
