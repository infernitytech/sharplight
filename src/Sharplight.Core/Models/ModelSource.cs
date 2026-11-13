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

    private static T ConvertValue<T>(object v, string key)
    {
        if (v is T t)
        {
            return t;
        }

        var target = typeof(T);
        if (target.IsArray && v is Array src)
        {
            var et = target.GetElementType()!;
            var dst = Array.CreateInstance(et, src.Length);
            for (int i = 0; i < src.Length; i++)
            {
                dst.SetValue(Convert.ChangeType(src.GetValue(i), et, CultureInfo.InvariantCulture), i);
            }

            return (T)(object)dst;
        }
        try
        {
            return (T)Convert.ChangeType(v, target, CultureInfo.InvariantCulture);
        }
        catch (Exception e)
        {
            throw new InvalidCastException($"Metadata '{key}' ({v.GetType().Name}) cannot be read as {target.Name}.", e);
        }
    }
}
