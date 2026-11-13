using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// Everything architecture-specific about importing one Hugging Face <c>model_type</c>: the canonical architecture id,
/// the hyper-parameter map and the weight name map including value/layout transforms. Adding a model family
/// is one class implementing this interface plus a <see cref="HfArchitectureMappings.Register"/> call.
/// </summary>
internal interface IHfArchitectureMapping
{
    /// <summary>Canonical architecture id (e.g. "qwen3_5").</summary>
    public string Architecture { get; }

    /// <summary>HF <c>model_type</c> values handled.</summary>
    public IReadOnlyList<string> ModelTypes { get; }

    /// <summary>Canonical <c>{arch}.*</c> hyper-parameters (full keys, including the architecture prefix).</summary>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig config, HfImportOptions options);

    /// <summary>
    /// Canonical weights. Unchanged tensors must be zero-copy views (<see cref="SafetensorsTensor.ToWeightInfo"/> + <c>As</c>);
    /// transformed ones lazy (<see cref="WeightInfo.Derived"/>). Tensors that belong to other models/sources
    /// (vision tower, ...) are skipped; tensors that are neither mapped nor skipped must throw.
    /// </summary>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig config, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HfImportOptions options);
}
