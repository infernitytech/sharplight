namespace Sharplight.Studio.Services.Compatibility;

/// <summary>The likely compatibility of a model, with the reason in a sentence.</summary>
/// <param name="State">The verdict.</param>
/// <param name="Subject">The architecture or weight format the verdict names (e.g. <c>hrm_text</c>, <c>Q8_K</c>), or null.</param>
/// <param name="Reason">Why, in one or two sentences.</param>
/// <param name="WeightsChecked">
/// The weight formats were read too (a GGUF tensor table, a checkpoint's quantization settings); false when only the architecture is
/// known so far.
/// </param>
internal sealed record ModelCompatibility(CompatibilityState State, string? Subject, string Reason, bool WeightsChecked = true)
{
    /// <summary>The verdict as a badge: <c>Supported</c>, <c>Unsupported architecture hrm_text</c>, ...</summary>
    public string Text => State switch
    {
        CompatibilityState.Supported => "Supported",
        CompatibilityState.Partial => "Partially supported",
        CompatibilityState.UnsupportedArchitecture => $"Unsupported architecture {Subject}",
        CompatibilityState.UnsupportedWeightFormat => $"Unsupported weight format {Subject}",
        CompatibilityState.Projector => "Projector",
        CompatibilityState.CheckedAtImport => "Checked at import",
        _ => "Unknown",
    };

    /// <summary>Importing or loading is expected to fail.</summary>
    public bool IsUnsupported => State is CompatibilityState.UnsupportedArchitecture or CompatibilityState.UnsupportedWeightFormat;

    /// <summary>A verdict that the metadata could not be read.</summary>
    /// <param name="reason">Why.</param>
    public static ModelCompatibility Unknown(string reason) => new(CompatibilityState.Unknown, null, reason, WeightsChecked: false);
}
