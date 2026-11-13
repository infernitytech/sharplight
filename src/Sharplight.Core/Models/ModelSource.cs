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
    /// <summary>Canonical metadata (see <see cref="CanonicalKeys"/>); values are scalars, strings or arrays of those.</summary>
    public required IReadOnlyDictionary<string, object> Metadata { get; init; }
    /// <summary>
    /// Weights by canonical name. Their bytes may be lazy (memory-mapped or derived); nothing is loaded until read.
    /// </summary>
    public required IReadOnlyDictionary<string, WeightInfo> Weights { get; init; }
    /// <summary>
    /// Resources (memory maps, file handles) that back <see cref="Weights"/>; the owner of the source disposes them once the model no longer needs the weights.
    /// </summary>
    public IReadOnlyList<IDisposable> OwnedResources { get; init; } = [];
    /// <summary>
    /// Format-specific payload for adapters that understand it — e.g. the parsed ONNX graph, which a
    /// generic ONNX adapter translates op-by-op into IR. Weight-only formats (GGUF, safetensors) leave it null
    /// and rely on an architecture adapter instead.
    /// </summary>
    public object? FormatData { get; init; }
    /// <summary>File or directory the source was opened from, when it came from one.</summary>
    public string? Path { get; init; }
    /// <summary>
    /// An already built definition (Sharplight IR), for formats that store one (saved model packages). Adapters are
    /// then not involved: <see cref="ModelAdapterRegistry.Build"/> returns it as is.
    /// </summary>
    public ModelDefinition? Definition { get; init; }
    /// <summary>Compilation caches stored with the model (saved compiled models), per device target.</summary>
    public Runtime.ICompilationCacheStore? CompilationCaches { get; init; }

    /// <summary>Whether the metadata contains <paramref name="key"/>.</summary>
    public bool Has(string key) => Metadata.ContainsKey(key);

    /// <summary>
    /// Reads a metadata value, converting numeric values and arrays to <typeparamref name="T"/>.
    /// </summary>
    /// <param name="key">Full metadata key.</param>
    /// <exception cref="KeyNotFoundException">The key is absent.</exception>
    /// <exception cref="InvalidCastException">The value cannot be converted to <typeparamref name="T"/>.</exception>
    public T Get<T>(string key) => Metadata.TryGetValue(key, out var v)
        ? ValueConverter.Convert<T>(v, "Metadata", key) : throw new KeyNotFoundException($"Missing metadata key '{key}'.");

    /// <summary>
    /// Reads an optional metadata value, converting like <see cref="Get{T}(string)"/>.
    /// </summary>
    /// <param name="key">Full metadata key.</param>
    /// <param name="defaultValue">Returned when the key is absent.</param>
    /// <exception cref="InvalidCastException">The value cannot be converted to <typeparamref name="T"/>.</exception>
    public T Get<T>(string key, T defaultValue) => Metadata.TryGetValue(key, out var v) ? ValueConverter.Convert<T>(v, "Metadata", key) : defaultValue;

    /// <summary>Reads <c>{Architecture}.{key}</c>.</summary>
    public T Arch<T>(string key) => Get<T>($"{Architecture}.{key}");
    /// <summary>Reads <c>{Architecture}.{key}</c>, or <paramref name="defaultValue"/> when absent.</summary>
    public T Arch<T>(string key, T defaultValue) => Get($"{Architecture}.{key}", defaultValue);

    /// <summary>Looks up a weight by canonical name; <see langword="null"/> when the model has none.</summary>
    public WeightInfo? Weight(string name) => Weights.TryGetValue(name, out var w) ? w : null;
}
