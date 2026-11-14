namespace Sharplight.Studio.Services.Compatibility;

/// <summary>Whether a model is likely to import and run, as far as its metadata tells before it is downloaded.</summary>
internal enum CompatibilityState
{
    /// <summary>The metadata could not be read, or it does not settle the question (e.g. an unknown quantization method).</summary>
    Unknown,

    /// <summary>The architecture is mapped and the weight formats have kernels.</summary>
    Supported,

    /// <summary>The architecture is mapped with a documented restriction.</summary>
    Partial,

    /// <summary>No importer mapping for the architecture.</summary>
    UnsupportedArchitecture,

    /// <summary>The weights use a quantization type or method without a reader or kernels.</summary>
    UnsupportedWeightFormat,

    /// <summary>
    /// The architecture is mapped, but this model uses a variant of it the importer rejects: a hyper-parameter or feature, or tensors
    /// without a canonical mapping.
    /// </summary>
    UnsupportedVariant,

    /// <summary>A multimodal projector (mmproj): loaded with its language model, not a model of its own.</summary>
    Projector,

    /// <summary>An ONNX model: its operators are checked when it is imported.</summary>
    CheckedAtImport,
}
