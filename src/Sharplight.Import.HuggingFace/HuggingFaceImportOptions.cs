using Sharplight.Import.PyTorch;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Options for the Hugging Face importer; configured with <c>services.AddHuggingFaceImporter(options => ...)</c> or
/// Microsoft.Extensions.Options.
/// </summary>
public sealed class HuggingFaceImportOptions
{
    /// <summary>
    /// Map multi-token-prediction (<c>mtp.*</c>) weights to canonical <c>mtp.*</c> names and report
    /// <c>mtp.layer_count</c>. Disable to get exactly the trunk, e.g. to mirror a GGUF converted without MTP.
    /// Default: true.
    /// </summary>
    public bool IncludeMtp { get; set; } = true;

    /// <summary>
    /// How <c>pytorch_model*.bin</c> shards are unpickled (allowed module prefixes, size limits). When the importer
    /// comes from the container, this is the <see cref="TorchCheckpointOptions"/> configured with <c>services.AddPyTorchImporter(options => ...)</c>.
    /// </summary>
    public TorchCheckpointOptions PyTorch { get; set; } = new();

    /// <summary>The default options (shared; never modified).</summary>
    internal static HuggingFaceImportOptions Default { get; } = new();
}
