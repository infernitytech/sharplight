namespace Sharplight.Llm.Serving;

/// <summary>
/// Computes the embedding rows of prompt media (images, audio clips) for the <see cref="InferenceEngine"/>, which runs it as a stage
/// before prefill: every content a step's prompt chunks need and that is not cached yet is encoded in one call per encoder, and the
/// rows are cached by <see cref="MediaContent.Hash"/>. Called on the engine's scheduler thread only.
/// </summary>
public interface IMediaEncoder
{
    /// <summary>Width of one embedding row (the decoder's <see cref="IBatchDecoder.EmbeddingWidth"/>).</summary>
    public int EmbeddingWidth { get; }

    /// <summary>Encodes <paramref name="contents"/> (all created for this encoder) and returns, in order, each one's rows [Rows, <see cref="EmbeddingWidth"/>].</summary>
    /// <param name="contents">The contents to encode.</param>
    public IReadOnlyList<float[]> Encode(IReadOnlyList<MediaContent> contents);
}
