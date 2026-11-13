using System.Collections.Concurrent;
using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>Options for the Hugging Face importer.</summary>
public sealed class HfImportOptions
{
    /// <summary>
    /// Map multi-token-prediction (<c>mtp.*</c>) weights to canonical <c>mtp.*</c> names and report
    /// <c>mtp.layer_count</c>. Disable to get exactly the trunk, e.g. to mirror a GGUF converted without MTP.
    /// </summary>
    public bool IncludeMtp { get; init; } = true;

    public static HfImportOptions Default { get; } = new();
}

/// <summary>
/// Everything architecture-specific about importing one Hugging Face <c>model_type</c>: the canonical architecture id,
/// the hyper-parameter map and the weight name map including value/layout transforms. Adding a model family
/// is one class implementing this interface plus a <see cref="HfArchitectureMappings.Register"/> call.
/// </summary>
public interface IHfArchitectureMapping
{
    /// <summary>Canonical architecture id (e.g. "qwen3_5").</summary>
    string Architecture { get; }

    /// <summary>HF <c>model_type</c> values handled.</summary>
    IReadOnlyList<string> ModelTypes { get; }

    /// <summary>Canonical <c>{arch}.*</c> hyper-parameters (full keys, including the architecture prefix).</summary>
    IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig config, HfImportOptions options);

    /// <summary>
    /// Canonical weights. Unchanged tensors must be zero-copy views (<see cref="SafetensorsTensor.ToWeightInfo"/> + <c>As</c>);
    /// transformed ones lazy (<see cref="WeightInfo.Derived"/>). Tensors that belong to other models/sources
    /// (vision tower, ...) are skipped; tensors that are neither mapped nor skipped must throw.
    /// </summary>
    IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig config, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HfImportOptions options);
}

/// <summary>Registry of <see cref="IHfArchitectureMapping"/>s keyed by HF <c>model_type</c>.</summary>
public static class HfArchitectureMappings
{
    static readonly ConcurrentDictionary<string, IHfArchitectureMapping> s_byModelType = new(StringComparer.OrdinalIgnoreCase);

    static HfArchitectureMappings()
    {
        Register(new Qwen35HfMapping());
    }

    public static void Register(IHfArchitectureMapping mapping)
    {
        foreach (var t in mapping.ModelTypes) s_byModelType[t] = mapping;
    }

    public static IEnumerable<IHfArchitectureMapping> All => s_byModelType.Values.Distinct();

    public static IHfArchitectureMapping? Find(string? modelType)
        => modelType != null && s_byModelType.TryGetValue(modelType, out var m) ? m : null;
}
