namespace Sharplight.Import.WhisperCpp;

/// <summary>The hyper-parameters at the start of a whisper.cpp model file.</summary>
/// <param name="VocabSize">Vocabulary size including the special and timestamp tokens.</param>
/// <param name="AudioContext">Encoder rows per window (1500).</param>
/// <param name="AudioWidth">Encoder width.</param>
/// <param name="AudioHeads">Encoder attention heads.</param>
/// <param name="AudioLayers">Encoder layers.</param>
/// <param name="TextContext">Decoder context in tokens (448).</param>
/// <param name="TextWidth">Decoder width.</param>
/// <param name="TextHeads">Decoder attention heads.</param>
/// <param name="TextLayers">Decoder layers.</param>
/// <param name="MelBins">Mel bins of the features.</param>
/// <param name="FileType">ggml file type (with the quantization version factor removed).</param>
internal sealed record WhisperCppHyperParameters(int VocabSize, int AudioContext, int AudioWidth, int AudioHeads, int AudioLayers,
    int TextContext, int TextWidth, int TextHeads, int TextLayers, int MelBins, int FileType)
{
    /// <summary>True for the multilingual vocabulary (language tokens after the start token).</summary>
    public bool IsMultilingual => VocabSize >= 51865;

    /// <summary>Language tokens of the vocabulary (99, or 100 for large-v3 with Cantonese).</summary>
    public int LanguageCount => VocabSize - 51765 - (IsMultilingual ? 1 : 0);
}
