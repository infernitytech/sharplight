using System.Diagnostics;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tokenizers;
using Sharplight.Vision;

namespace Sharplight.Llm;

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
    public InferenceSession CreateSession(string graph = WellKnown.DecodeGraph, long? memoryLimit = null)
    {
        var sw = Stopwatch.StartNew();
        var compiled = CompiledModel.Compile(Definition.Graph(graph), new PlacementOptions
        {
            Devices = DevicesWithLimit(memoryLimit),
            Fallback = _options.Fallback,
            ReservePerDevice = _options.ReservePerDevice ?? (memoryLimit is null ? DefaultReserve : 0),
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
    public Serving.IBatchDecoder CreateBatchDecoder(int slots, int? contextLength = null, long? memoryLimit = null)
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
        var dec = Serving.MultiSequenceDecoder.Create(graph, slots, ctx, _options.MaxBatchTokens, new PlacementOptions
        {
            Devices = DevicesWithLimit(memoryLimit),
            Fallback = _options.Fallback,
            ReservePerDevice = _options.ReservePerDevice ?? (memoryLimit is null ? DefaultReserve : 0),
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

    private const long DefaultReserve = 1L << 30;

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

    public void Dispose()
    {
        foreach (var r in Definition.OwnedResources)
        {
            r.Dispose();
        }
    }
}
