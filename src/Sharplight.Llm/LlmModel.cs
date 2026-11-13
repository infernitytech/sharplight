using System.Diagnostics;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tokenizers;
using Sharplight.Vision;

namespace Sharplight.Llm;

/// <summary>
/// High-level entry point: a loaded decoder-style language model (weights, tokenizer and chat template). Loading reads the
/// model file and builds its compute graphs; nothing is compiled or uploaded to a device until a session, batch decoder or
/// vision encoder is created from it. From a model you create
/// <list type="bullet">
/// <item><description>an <see cref="InferenceSession"/> plus a <see cref="Generator"/> for simple single-sequence generation,</description></item>
/// <item><description>a batch decoder (<see cref="CreateBatchDecoder"/>) for the continuous-batching <see cref="Serving.InferenceEngine"/>,</description></item>
/// <item><description>a <see cref="VisionEncoder"/> (<see cref="CreateVisionEncoder"/>) for image prompts.</description></item>
/// </list>
/// The model owns the weight memory: dispose it after everything created from it has been disposed. Creating sessions and
/// decoders is thread-safe; the created objects are not.
/// </summary>
/// <example>
/// <code>
/// GgufImporter.Register();   // from Sharplight.Gguf; the host registers the formats it supports
/// using var model = LlmModel.Load("qwen3-4b-q4_k_m.gguf", new LlmLoadOptions { Devices = [new DeviceSlot { Device = gpu }] });
/// using var session = model.CreateSession();
/// var generator = new Generator(session, model.Tokenizer);
/// var options = new GenerationOptions { MaxNewTokens = 128, Sampling = SamplingOptions.Greedy };
/// foreach (var token in generator.Generate("The capital of France is", options))
/// {
///     Console.Write(token.Text);
/// }
/// </code>
/// </example>
public sealed class LlmModel : IDisposable
{
    static LlmModel() => BuiltinAdapters.RegisterAll();

    /// <summary>The built model: architecture, hyper-parameters, weights and compute graphs.</summary>
    public ModelDefinition Definition { get; }
    /// <summary>The canonical model source the definition was built from (what a saved package stores).</summary>
    public ModelSource Source { get; }
    /// <summary>The model's tokenizer (shared; safe for concurrent use).</summary>
    public ITokenizer Tokenizer { get; }
    /// <summary>The Jinja chat template stored in the model file (<c>tokenizer.chat_template</c>), or null if it has none.</summary>
    public string? ChatTemplate { get; }
    private readonly LlmLoadOptions _options;
    private readonly Dictionary<string, (CompilationCache Cache, List<IComputeDevice> Devices)> _caches = [];
    private readonly Lock _cacheLock = new();

    private LlmModel(ModelDefinition def, ModelSource source, ITokenizer tok, LlmLoadOptions options)
    {
        Definition = def;
        Source = source;
        Tokenizer = tok;
        _options = options;
        ChatTemplate = def.SourceMetadata.TryGetValue(CanonicalKeys.ChatTemplate, out var t) ? t as string : null;
    }

    /// <summary>
    /// Loads a model through <see cref="ModelImporterRegistry"/>; the host must have registered an
    /// importer for the file's format (e.g. <c>GgufImporter.Register()</c> from Sharplight.Gguf).
    /// </summary>
    /// <param name="path">Model file (GGUF, ...) or directory (Hugging Face layout), as understood by the registered importers.</param>
    /// <param name="options">Devices and compile settings; also kept for later session / decoder creation.</param>
    /// <exception cref="NotSupportedException">No registered importer or architecture adapter handles the model.</exception>
    public static LlmModel Load(string path, LlmLoadOptions options)
    {
        var sw = Stopwatch.StartNew();
        var sources = ModelImporterRegistry.OpenSet(path, [.. options.Auxiliary]);
        var def = ModelAdapterRegistry.Build(sources, options.Adapter.ContextLength > 0 ? options.Adapter : options.Adapter.WithContext(options.ContextLength));
        var tok = TokenizerFactory.Create(TokenizerData.FromMetadata(def.SourceMetadata));
        options.Log?.Invoke($"Loaded {def.Name} ({def.Architecture}): {def.Weights.Count} tensors, {def.Weights.Values.Sum(w => w.ByteSize) / 1e9:F2} GB, vocab {tok.VocabSize} [{sw.ElapsedMilliseconds} ms]");
        return new LlmModel(def, sources.Primary, tok, options);
    }

    /// <summary>
    /// Compiles the decoder onto the configured devices and returns a fresh single-sequence session. The session owns device
    /// memory for its share of the weights, the KV cache and the activations.
    /// </summary>
    /// <param name="graph">The graph to compile: <see cref="WellKnown.DecodeGraph"/> (text) or <see cref="WellKnown.DecodeMultimodalGraph"/> (text and images).</param>
    /// <param name="memoryLimit">Caps the bytes of the primary device used by this session (null: the devices as configured).</param>
    public InferenceSession CreateSession(string graph = WellKnown.DecodeGraph, long? memoryLimit = null)
    {
        var sw = Stopwatch.StartNew();
        var placement = PlacementFor(memoryLimit, _options.ContextLength, rowBound: _options.MaxBatchTokens);
        var compiled = CompiledModel.Compile(WithSamplingHead(Definition.Graph(graph)), placement);
        _options.Log?.Invoke($"Compiled '{graph}' in {sw.ElapsedMilliseconds} ms\n{compiled.Plan.Describe()}");
        return new InferenceSession(compiled);
    }

    /// <summary>
    /// Creates the decoder the serving layer batches requests over: with <paramref name="slots"/> &gt; 1 and a graph whose
    /// stateful ops all have multi-sequence forms, a <see cref="Serving.MultiSequenceDecoder"/> (rows of up to
    /// <paramref name="slots"/> sequences per forward pass, each with <paramref name="contextLength"/> tokens of context,
    /// default the load option); otherwise a single-slot <see cref="Serving.SessionBatchDecoder"/>.
    /// </summary>
    /// <param name="slots">Sequences decoded together.</param>
    /// <param name="contextLength">Context per sequence (default <see cref="LlmLoadOptions.ContextLength"/>).</param>
    /// <param name="memoryLimit">Device bytes the decoder may use, or null for the devices as configured.</param>
    /// <param name="maxLogitsRows">
    /// Logits rows one step may return (default <paramref name="slots"/>, one per sequence). Speculative decoding needs
    /// <c>slots · (draft tokens + 1)</c>; each extra row costs <c>vocabulary · 4</c> bytes of activations.
    /// </param>
    public Serving.IBatchDecoder CreateBatchDecoder(int slots, int? contextLength = null, long? memoryLimit = null, int? maxLogitsRows = null)
    {
        var graph = Definition.Graph(WellKnown.DecodeGraph);
        string? why = null;
        if (slots <= 1 || !Graph.MultiSequence.IsSupported(graph, out why))
        {
            if (slots > 1)
            {
                _options.Log?.Invoke($"Multi-sequence decoding unavailable ({why}); serving one sequence at a time.");
            }

            return new Serving.SessionBatchDecoder(CreateSession(memoryLimit: memoryLimit));
        }
        int ctx = contextLength ?? _options.ContextLength;
        var sw = Stopwatch.StartNew();
        var placement = PlacementFor(memoryLimit, ctx, rowBound: Math.Max(slots, maxLogitsRows ?? 1));
        var dec = Serving.MultiSequenceDecoder.Create(graph, slots, ctx, _options.MaxBatchTokens, placement, UsesDeviceSampling);
        _options.Log?.Invoke($"Compiled '{WellKnown.DecodeGraph}' for {slots} sequences x {ctx} tokens in {sw.ElapsedMilliseconds} ms\n{dec.Compiled.Plan.Describe()}");
        return dec;
    }

    private const long DefaultReserve = 1L << 30;

    /// <summary>
    /// Whether decoders get a device sampling head: enabled by the options and the environment, and the logits are computed on an
    /// accelerator (the last device in pipeline order); on the CPU the head would save nothing over sampling the logits directly.
    /// </summary>
    private bool UsesDeviceSampling => _options.DeviceSampling && DiagnosticSwitches.DeviceSampling
        && _options.Devices.Count > 0 && _options.Devices[^1].Device.Info.Kind != DeviceKind.Cpu;

    private ComputeGraph WithSamplingHead(ComputeGraph graph)
        => UsesDeviceSampling && SamplingHead.CanAppend(graph) ? SamplingHead.Append(graph) : graph;

    /// <summary>Device placement and compile settings shared by sessions and batch decoders.</summary>
    /// <param name="memoryLimit">Bytes the primary device may hold, or null for the devices as configured.</param>
    /// <param name="contextLength">KV-cache rows per sequence.</param>
    /// <param name="rowBound">Upper bound of the <c>R</c> symbol (rows whose logits are read per step).</param>
    private PlacementOptions PlacementFor(long? memoryLimit, int contextLength, long rowBound) => new()
    {
        Devices = DevicesWithLimit(memoryLimit),
        Fallback = _options.Fallback,
        ReservePerDevice = _options.ReservePerDevice ?? (memoryLimit is null ? DefaultReserve : 0),
        Compile = new CompileOptions
        {
            MaxTokens = _options.MaxBatchTokens,
            ContextLength = contextLength,
            SymbolBounds = new Dictionary<string, long> { ["R"] = rowBound },
            DisabledFeatures = _options.DisabledFeatures,
            CacheFor = CacheFor,
        },
    };

    /// <summary>
    /// Caps the first (primary) device's share of the model at <paramref name="memoryLimit"/> bytes (layers beyond it go to
    /// the fallback device or the load fails); other devices keep their own limits. Null = the devices as configured.
    /// </summary>
    private IReadOnlyList<DeviceSlot> DevicesWithLimit(long? memoryLimit)
    {
        if (memoryLimit is not { } limit || _options.Devices.Count == 0)
        {
            return _options.Devices;
        }

        var d0 = _options.Devices[0];
        var first = new DeviceSlot { Device = d0.Device, Weight = d0.Weight, MemoryLimit = d0.MemoryLimit is { } m ? Math.Min(m, limit) : limit };
        return [first, .. _options.Devices.Skip(1)];
    }

    /// <summary>
    /// Device memory needed to serve this model with <paramref name="slots"/> concurrent sequences of
    /// <paramref name="contextLength"/> tokens (default the load option): weights + KV/recurrent state + activation bound.
    /// Computed from the graph alone; nothing is allocated on a device.
    /// </summary>
    public ModelFootprint EstimateFootprint(int slots = 1, int? contextLength = null)
    {
        int ctx = contextLength ?? _options.ContextLength;
        var graph = Definition.Graph(WellKnown.DecodeGraph);
        bool multi = slots > 1 && Graph.MultiSequence.IsSupported(graph, out _);
        return PlacementPlanner.Estimate(graph, new CompileOptions
        {
            MaxTokens = _options.MaxBatchTokens,
            ContextLength = ctx,
            SymbolBounds = new Dictionary<string, long> { ["R"] = multi ? slots : _options.MaxBatchTokens },
        }, multi ? slots : 1);
    }

    /// <summary>True when the model has a vision tower (loaded from a file containing one, or together with a "mmproj" auxiliary file).</summary>
    public bool SupportsImages => Definition.Graphs.ContainsKey(WellKnown.VisionEncodeGraph);

    /// <summary>
    /// Compiles the vision encoder onto the configured devices. The returned encoder turns <see cref="ImageInput"/>s into
    /// embeddings; pass it to a <see cref="Generator"/> (with a session created on <see cref="WellKnown.DecodeMultimodalGraph"/>).
    /// </summary>
    public VisionEncoder CreateVisionEncoder()
    {
        var sw = Stopwatch.StartNew();
        var enc = VisionEncoder.Create(Definition, _options.Devices, _options.Fallback, _options.ImageTokens, _options.DisabledFeatures, CacheFor);
        _options.Log?.Invoke($"Compiled '{WellKnown.VisionEncodeGraph}' (up to {enc.MaxPatches / enc.Parameters.MergeUnit} image tokens) in {sw.ElapsedMilliseconds} ms\n{enc.Compiled.Plan.Describe()}");
        return enc;
    }

    /// <summary>One recording cache per device target, seeded from caches stored in the model file.</summary>
    private ICompilationCache CacheFor(IComputeDevice device)
    {
        var target = device.CompilationTarget;
        lock (_cacheLock)
        {
            if (!_caches.TryGetValue(target, out var e))
            {
                _caches[target] = e = (new CompilationCache(target, Source.CompilationCaches?.Find(target)), []);
            }

            if (!e.Devices.Contains(device))
            {
                e.Devices.Add(device);
            }

            return e.Cache;
        }
    }

    /// <summary>
    /// Compilation artifacts of every graph compiled so far (call after <see cref="CreateSession"/>), one cache per
    /// device target, including device state such as driver pipeline caches. Save them with the model source
    /// (e.g. Sharplight.Package's <c>ModelPackage.Save</c>) so later loads skip kernel compilation.
    /// </summary>
    public IReadOnlyList<CompilationCache> ExportCompilation()
    {
        lock (_cacheLock)
        {
            foreach (var (cache, devices) in _caches.Values)
            {
                foreach (var d in devices)
                {
                    d.ExportCompilationState(cache);
                }
            }

            return [.. _caches.Values.Select(e => e.Cache)];
        }
    }

    /// <summary>Releases the weight memory (mapped files, host buffers). Sessions and decoders created from the model must be disposed first.</summary>
    public void Dispose()
    {
        foreach (var r in Definition.OwnedResources)
        {
            r.Dispose();
        }
    }
}
