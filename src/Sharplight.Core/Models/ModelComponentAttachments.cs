namespace Sharplight.Models;

/// <summary>
/// Well-known local attachment names of components (<see cref="ModelComponent.Attachments"/>), read by the Core task layer. Open-ended like
/// roles: a component may carry any other attachment its task implementation knows.
/// </summary>
public static class ModelComponentAttachments
{
    /// <summary>
    /// A tokenizer stored as <see cref="Tokenizers.TokenizerData"/> JSON (<see cref="Tokenizers.TokenizerData.ToJson"/>); preferred over
    /// metadata keys by <c>TaskModel.CreateTokenizer</c>.
    /// </summary>
    public const string Tokenizer = "tokenizer";

    /// <summary>A mel filter bank of a log-mel feature extractor: little-endian f32 [mel bins, FFT size / 2 + 1].</summary>
    public const string MelFilters = "mel-filters";
}
