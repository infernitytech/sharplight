using System.Diagnostics;
using Sharplight.Architectures;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tokenizers;

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
        var def = ModelAdapterRegistry.Build(sources, options.Adapter);
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
        ModelAdapterRegistry.Register(new DecoderAdapter());
    }
}
