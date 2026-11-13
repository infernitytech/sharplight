using System.Diagnostics;
using Sharplight.Architectures;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tokenizers;
using Sharplight.Vision;

namespace Sharplight.Llm;

/// <summary>Options for loading and compiling a language model.</summary>
public sealed class LlmLoadOptions
{
    /// <summary>Devices in pipeline order (layers are split across them). Required.</summary>
    public required IReadOnlyList<DeviceSlot> Devices { get; init; }
    /// <summary>Fallback device (e.g. CPU) for layers that don't fit and unsupported ops.</summary>
    public DeviceSlot? Fallback { get; init; }
    public int ContextLength { get; init; } = 8192;
    /// <summary>Max tokens per forward pass (prefill chunk size).</summary>
    public int MaxBatchTokens { get; init; } = 512;
    public AdapterOptions Adapter { get; init; } = new();
    /// <summary>Device features (<see cref="DeviceFeatures"/>) not to use when compiling this model, e.g. "matrix_f16".</summary>
    public IReadOnlySet<string> DisabledFeatures { get; init; } = new HashSet<string>();
    /// <summary>Auxiliary model files, e.g. ("mmproj", path) for a vision projector.</summary>
    public IReadOnlyList<(string Role, string Path)> Auxiliary { get; init; } = [];
    /// <summary>
    /// Image token budget (output tokens per image): the vision encoder is compiled for at most
    /// <c>MaxTokens</c> (default 1024) tokens per image and images are resized to fit; <c>MinTokens</c> upsizes small images.
    /// </summary>
    public ImageTokenLimits? ImageTokens { get; init; }
    public Action<string>? Log { get; init; }
}

/// <summary>
/// High-level entry point: a loaded model definition plus its tokenizer and chat template.
/// Create <see cref="InferenceSession"/>s / <see cref="Generator"/>s from it.
/// </summary>
public sealed class LlmModel : IDisposable
{
    static LlmModel() => BuiltinAdapters.RegisterAll();

    public ModelDefinition Definition { get; }
    /// <summary>The canonical model source the definition was built from (what a saved package stores).</summary>
    public ModelSource Source { get; }
    public ITokenizer Tokenizer { get; }
    public string? ChatTemplate { get; }
    readonly LlmLoadOptions _options;
    readonly Dictionary<string, (CompilationCache Cache, List<IComputeDevice> Devices)> _caches = [];
    readonly Lock _cacheLock = new();

    LlmModel(ModelDefinition def, ModelSource source, ITokenizer tok, LlmLoadOptions options)
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
    public static LlmModel Load(string path, LlmLoadOptions options)
    {
        var sw = Stopwatch.StartNew();
        var sources = ModelImporterRegistry.OpenSet(path, [.. options.Auxiliary]);
        var def = ModelAdapterRegistry.Build(sources, options.Adapter.ContextLength > 0 ? options.Adapter : options.Adapter.WithContext(options.ContextLength));
        var tok = TokenizerFactory.Create(TokenizerData.FromMetadata(def.SourceMetadata));
        options.Log?.Invoke($"Loaded {def.Name} ({def.Architecture}): {def.Weights.Count} tensors, {def.Weights.Values.Sum(w => w.ByteSize) / 1e9:F2} GB, vocab {tok.VocabSize} [{sw.ElapsedMilliseconds} ms]");
        return new LlmModel(def, sources.Primary, tok, options);
    }

    /// <summary>Compiles the decoder onto the configured devices and returns a fresh session.</summary>
    public InferenceSession CreateSession(string graph = WellKnown.DecodeGraph)
    {
        var sw = Stopwatch.StartNew();
        var compiled = CompiledModel.Compile(Definition.Graph(graph), new PlacementOptions
        {
            Devices = _options.Devices,
            Fallback = _options.Fallback,
            Compile = new CompileOptions
            {
                MaxTokens = _options.MaxBatchTokens,
                ContextLength = _options.ContextLength,
                SymbolBounds = new Dictionary<string, long> { ["R"] = _options.MaxBatchTokens },
                DisabledFeatures = _options.DisabledFeatures,
                CacheFor = CacheFor,
            },
        });
        _options.Log?.Invoke($"Compiled '{graph}' in {sw.ElapsedMilliseconds} ms\n{compiled.Plan.Describe()}");
        return new InferenceSession(Definition, compiled);
    }

    /// <summary>
    /// Creates the decoder the serving layer batches requests over: with <paramref name="slots"/> &gt; 1 and a graph whose
    /// stateful ops all have multi-sequence forms, a <see cref="Serving.MultiSequenceDecoder"/> (rows of up to
    /// <paramref name="slots"/> sequences per forward pass, each with <paramref name="contextLength"/> tokens of context,
    /// default the load option); otherwise a single-slot <see cref="Serving.SessionBatchDecoder"/>.
    /// </summary>
    public Serving.IBatchDecoder CreateBatchDecoder(int slots, int? contextLength = null)
    {
        var graph = Definition.Graph(WellKnown.DecodeGraph);
        string? why = null;
        if (slots <= 1 || !Graph.MultiSequence.IsSupported(graph, out why))
        {
            if (slots > 1) _options.Log?.Invoke($"Multi-sequence decoding unavailable ({why}); serving one sequence at a time.");
            return new Serving.SessionBatchDecoder(CreateSession());
        }
        int ctx = contextLength ?? _options.ContextLength;
        var sw = Stopwatch.StartNew();
        var dec = Serving.MultiSequenceDecoder.Create(graph, slots, ctx, _options.MaxBatchTokens, new PlacementOptions
        {
            Devices = _options.Devices,
            Fallback = _options.Fallback,
            Compile = new CompileOptions
            {
                MaxTokens = _options.MaxBatchTokens,
                ContextLength = ctx,
                SymbolBounds = new Dictionary<string, long> { ["R"] = Math.Max(slots, 1) },
                DisabledFeatures = _options.DisabledFeatures,
                CacheFor = CacheFor,
            },
        });
        _options.Log?.Invoke($"Compiled '{WellKnown.DecodeGraph}' for {slots} sequences x {ctx} tokens in {sw.ElapsedMilliseconds} ms\n{dec.Compiled.Plan.Describe()}");
        return dec;
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
    ICompilationCache CacheFor(IComputeDevice device)
    {
        var target = device.CompilationTarget;
        lock (_cacheLock)
        {
            if (!_caches.TryGetValue(target, out var e))
                _caches[target] = e = (new CompilationCache(target, Source.CompilationCaches?.Find(target)), []);
            if (!e.Devices.Contains(device)) e.Devices.Add(device);
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
                foreach (var d in devices) d.ExportCompilationState(cache);
            return [.. _caches.Values.Select(e => e.Cache)];
        }
    }

    public void Dispose()
    {
        foreach (var r in Definition.OwnedResources) r.Dispose();
    }
}

/// <summary>Registers the architecture adapters shipped with Sharplight.</summary>
public static class BuiltinAdapters
{
    static int s_done;
    public static void RegisterAll()
    {
        if (Interlocked.Exchange(ref s_done, 1) == 1) return;
        ModelAdapterRegistry.Register(new Qwen35Adapter());
        Sharplight.Vision.MultimodalRegistry.Register(new Qwen35Multimodal());
        ModelAdapterRegistry.Register(new DecoderAdapter());
    }
}
