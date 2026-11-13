using Sharplight.Models;

namespace Sharplight.Gguf;

/// <summary>
/// Everything architecture-specific about importing one GGUF (llama.cpp) architecture: the canonical
/// architecture id, the hyper-parameter map and the tensor name map. llama.cpp's naming stays inside this
/// assembly; adding a model family is one class implementing this interface plus a
/// <see cref="GgufArchitectureMappings.Register"/> call.
/// </summary>
internal interface IGgufArchitectureMapping
{
    /// <summary>GGUF <c>general.architecture</c> values handled (e.g. "qwen35").</summary>
    public IReadOnlyList<string> GgufArchitectures { get; }

    /// <summary>Canonical architecture id (e.g. "qwen3_5").</summary>
    public string Architecture { get; }

    /// <summary>Canonical <c>{arch}.*</c> hyper-parameters (full keys) from the GGUF metadata.</summary>
    public IReadOnlyDictionary<string, object> MapHyperParameters(GgufMetadataView gguf);

    /// <summary>Canonical name of a GGUF tensor, or null when it has none (the import then fails).</summary>
    public string? MapWeightName(string ggufName, GgufMetadataView gguf);

    /// <summary>
    /// Optional value/layout transform of a tensor that already carries its canonical name (e.g. a lazily derived row
    /// permutation), so canonical weights are format independent. Default: unchanged (zero-copy).
    /// </summary>
    public WeightInfo MapWeight(WeightInfo weight, GgufMetadataView gguf) => weight;

    /// <summary>Canonical weights that GGUF does not store but the metadata determines (e.g. computed rope frequency factors).</summary>
    public IEnumerable<WeightInfo> SyntheticWeights(GgufMetadataView gguf, IReadOnlyDictionary<string, WeightInfo> weights) => [];

    /// <summary>
    /// Last chance to adjust the canonical hyper-parameters once the canonical weights are known, for architectures whose layout is only
    /// visible from the tensors present (e.g. a missing MLP norm means attention and MLP run in parallel). Default: no change.
    /// </summary>
    public void RefineHyperParameters(Dictionary<string, object> hyperParameters, GgufMetadataView gguf, IReadOnlyDictionary<string, WeightInfo> weights)
    {
    }

    /// <summary>
    /// The canonical architecture id for this particular file. Mappings whose GGUF architecture covers several
    /// layouts (e.g. <c>clip</c> projector types) override it; default: <see cref="Architecture"/>.
    /// </summary>
    public string ResolveArchitecture(GgufMetadataView gguf) => Architecture;
}
