using System.Collections.Concurrent;
using System.Globalization;

namespace Sharplight.Models;

/// <summary>
/// Raw, format-neutral view of a model file (or a set of files) before interpretation:
/// metadata key/values (<see cref="CanonicalKeys"/>) and canonically named weights.
/// Produced by <see cref="IModelImporter"/>s (GGUF, Hugging Face checkpoints, ONNX, ...).
/// </summary>
public sealed class ModelSource
{
    /// <summary>Container format the source was imported from (informational only, e.g. "gguf", "hf").</summary>
    public required string Format { get; init; }
    /// <summary>Canonical architecture id (e.g. "qwen3_5", "llama", "clip"); importers map their own ids to it.</summary>
    public required string Architecture { get; init; }
    public required IReadOnlyDictionary<string, object> Metadata { get; init; }
    public required IReadOnlyDictionary<string, WeightInfo> Weights { get; init; }
    public IReadOnlyList<IDisposable> OwnedResources { get; init; } = [];
    /// <summary>
    /// Format-specific payload for adapters that understand it — e.g. the parsed ONNX graph, which a
    /// generic ONNX adapter translates op-by-op into IR. Weight-only formats (GGUF, safetensors) leave it null
    /// and rely on an architecture adapter instead.
    /// </summary>
    public object? FormatData { get; init; }
    public string? Path { get; init; }
    /// <summary>
    /// An already built definition (Sharplight IR), for formats that store one (saved model packages). Adapters are
    /// then not involved: <see cref="ModelAdapterRegistry.Build"/> returns it as is.
    /// </summary>
    public ModelDefinition? Definition { get; init; }
    /// <summary>Compilation caches stored with the model (saved compiled models), per device target.</summary>
    public Runtime.ICompilationCacheStore? CompilationCaches { get; init; }

    public bool Has(string key) => Metadata.ContainsKey(key);

    public T Get<T>(string key) => Metadata.TryGetValue(key, out var v)
        ? ConvertValue<T>(v, key) : throw new KeyNotFoundException($"Missing metadata key '{key}'.");

    public T Get<T>(string key, T defaultValue) => Metadata.TryGetValue(key, out var v) ? ConvertValue<T>(v, key) : defaultValue;

    /// <summary>Reads <c>{Architecture}.{key}</c>.</summary>
    public T Arch<T>(string key) => Get<T>($"{Architecture}.{key}");
    public T Arch<T>(string key, T defaultValue) => Get($"{Architecture}.{key}", defaultValue);

    public WeightInfo? Weight(string name) => Weights.TryGetValue(name, out var w) ? w : null;

    static T ConvertValue<T>(object v, string key)
    {
        if (v is T t) return t;
        var target = typeof(T);
        if (target.IsArray && v is Array src)
        {
            var et = target.GetElementType()!;
            var dst = Array.CreateInstance(et, src.Length);
            for (int i = 0; i < src.Length; i++) dst.SetValue(Convert.ChangeType(src.GetValue(i), et, CultureInfo.InvariantCulture), i);
            return (T)(object)dst;
        }
        try { return (T)Convert.ChangeType(v, target, CultureInfo.InvariantCulture); }
        catch (Exception e) { throw new InvalidCastException($"Metadata '{key}' ({v.GetType().Name}) cannot be read as {target.Name}.", e); }
    }
}

/// <summary>A primary model source plus optional auxiliary sources (e.g. a vision projector "mmproj").</summary>
public sealed class ModelSourceSet
{
    public required ModelSource Primary { get; init; }
    public IReadOnlyDictionary<string, ModelSource> Auxiliary { get; init; } = new Dictionary<string, ModelSource>();
}

/// <summary>Options passed to adapters.</summary>
public sealed class AdapterOptions
{
    /// <summary>Data type of attention KV caches.</summary>
    public Tensors.DataType KvCacheType { get; init; } = Tensors.DataType.F16;
    /// <summary>Emit auxiliary graphs (multi-token prediction heads, embedding-input decode, ...).</summary>
    public bool IncludeAuxiliaryGraphs { get; init; } = true;
    /// <summary>
    /// Context length the model will run with (0 = unknown, use the model's maximum). Lets an adapter make context-dependent
    /// choices, e.g. longrope's short vs. long frequency factors.
    /// </summary>
    public int ContextLength { get; init; }
    /// <summary>Pooling of the <see cref="WellKnown.EmbedGraph"/> ("mean", "cls", "last", "none"); null = what the model declares.</summary>
    public string? EmbeddingPooling { get; init; }
    /// <summary>Whether <see cref="WellKnown.EmbedGraph"/> L2-normalizes; null = what the model declares.</summary>
    public bool? EmbeddingNormalize { get; init; }

    public AdapterOptions WithContext(int contextLength) => new()
    {
        KvCacheType = KvCacheType, IncludeAuxiliaryGraphs = IncludeAuxiliaryGraphs, ContextLength = contextLength,
        EmbeddingPooling = EmbeddingPooling, EmbeddingNormalize = EmbeddingNormalize,
    };
}

/// <summary>
/// Interprets a <see cref="ModelSource"/> of a particular architecture and emits the format-agnostic
/// <see cref="ModelDefinition"/> (weights + compute graphs). This is the only architecture-specific code.
/// </summary>
public interface IModelAdapter
{
    string Name { get; }
    /// <summary>Architecture ids handled (canonical ids, e.g. "qwen3_5").</summary>
    IReadOnlyList<string> Architectures { get; }
    bool CanLoad(ModelSource source) => Architectures.Contains(source.Architecture, StringComparer.OrdinalIgnoreCase);
    ModelDefinition Build(ModelSourceSet sources, AdapterOptions options);
}

/// <summary>Global adapter registry; adapters register themselves (or are registered by the host).</summary>
public static class ModelAdapterRegistry
{
    static readonly ConcurrentDictionary<string, IModelAdapter> s_adapters = new(StringComparer.OrdinalIgnoreCase);

    public static void Register(IModelAdapter adapter)
    {
        foreach (var a in adapter.Architectures) s_adapters[a] = adapter;
    }

    public static IModelAdapter? Find(ModelSource source)
        => s_adapters.TryGetValue(source.Architecture, out var a) && a.CanLoad(source) ? a
           : s_adapters.Values.Distinct().FirstOrDefault(x => x.CanLoad(source));

    public static IEnumerable<IModelAdapter> All => s_adapters.Values.Distinct();

    public static ModelDefinition Build(ModelSourceSet sources, AdapterOptions? options = null)
    {
        if (sources.Primary.Definition is { } prebuilt)
        {
            if (sources.Auxiliary.Count > 0)
                throw new NotSupportedException($"'{sources.Primary.Path}' already contains a built model; auxiliary files ({string.Join(", ", sources.Auxiliary.Keys)}) must be added before it is saved.");
            return prebuilt;
        }
        var adapter = Find(sources.Primary)
            ?? throw new NotSupportedException($"No adapter for architecture '{sources.Primary.Architecture}'. Registered: {string.Join(", ", s_adapters.Keys)}");
        return adapter.Build(sources, options ?? new AdapterOptions());
    }
}
