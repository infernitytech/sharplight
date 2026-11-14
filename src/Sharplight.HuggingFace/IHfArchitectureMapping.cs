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

    /// <summary>
    /// The hyper-parameters or features of this checkpoint that the mapping (or the adapter it maps to) does not run, as one reason each;
    /// none when it accepts the checkpoint. The importer rejects a checkpoint with reasons before mapping it, and
    /// <see cref="HfSupport.Validate"/> reports them from <c>config.json</c> before the checkpoint is downloaded, so a mapping names a
    /// variant it rejects here instead of throwing while it maps. Only the configuration may be read. Default: none.
    /// </summary>
    /// <param name="config">The checkpoint's configuration.</param>
    public IEnumerable<string> Validate(HfConfig config) => [];

    /// <summary>Canonical <c>{arch}.*</c> hyper-parameters (full keys, including the architecture prefix).</summary>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig config, HfImportOptions options);

    /// <summary>
    /// Canonical weights. Unchanged tensors must be zero-copy views (<see cref="SafetensorsTensor.ToWeightInfo"/> + <c>As</c>);
    /// transformed ones lazy (<see cref="WeightInfo.Derived"/>). Tensors that belong to other models/sources
    /// (vision tower, ...) are skipped; tensors that are neither mapped nor skipped must throw.
    /// </summary>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig config, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HfImportOptions options);

    /// <summary>
    /// What restricts one of <see cref="ModelTypes"/> when it runs only in part (as <c>docs/model-support.md</c> lists it as partial),
    /// or null when it is fully supported. Default: null.
    /// </summary>
    /// <param name="modelType">One of <see cref="ModelTypes"/>.</param>
    public string? PartialSupport(string modelType) => null;
}
