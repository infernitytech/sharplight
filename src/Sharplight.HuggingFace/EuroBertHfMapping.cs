namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>eurobert</c> (EuroBERT): Llama layers (RMS norms, rotary embedding, SwiGLU) with bidirectional attention, run as an
/// embedding model (canonical <c>attention.causal</c> = false; mean pooling unless sentence-transformers files declare another). The masked-LM
/// head is not used.
/// </summary>
internal sealed class EuroBertHfMapping : HfDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["eurobert"];

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, HfConfig c) => md["decoder.attention.causal"] = false;
}
