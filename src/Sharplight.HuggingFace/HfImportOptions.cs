namespace Sharplight.HuggingFace;

/// <summary>Options for the Hugging Face importer.</summary>
public sealed class HfImportOptions
{
    /// <summary>
    /// Map multi-token-prediction (<c>mtp.*</c>) weights to canonical <c>mtp.*</c> names and report
    /// <c>mtp.layer_count</c>. Disable to get exactly the trunk, e.g. to mirror a GGUF converted without MTP.
    /// Default: true.
    /// </summary>
    public bool IncludeMtp { get; init; } = true;

    /// <summary>The default options (a shared, immutable instance).</summary>
    public static HfImportOptions Default { get; } = new();
}
