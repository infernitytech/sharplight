namespace Sharplight.Llm.Serving;

/// <summary>
/// Consecutive rows of a <see cref="SequenceInput"/> whose input is an embedding (rows of a vision or audio encoder) instead of the
/// embedding of their token id.
/// </summary>
/// <param name="Row">First row of the input the embeddings fill.</param>
/// <param name="Values">
/// The rows, [rows, <see cref="IBatchDecoder.EmbeddingWidth"/>] row-major (the width includes DeepStack blocks where the model has them).
/// </param>
/// <param name="MediaWeights">
/// Whether the rows go through the language model's media weight set where it has one (<see cref="Models.InputRole.MediaWeightMask"/>, CogVLM's
/// visual experts); ignored by other models.
/// </param>
public readonly record struct EmbeddingRun(int Row, ReadOnlyMemory<float> Values, bool MediaWeights = true)
{
    /// <summary>Number of rows of width <paramref name="width"/> the run holds.</summary>
    /// <param name="width">The decoder's <see cref="IBatchDecoder.EmbeddingWidth"/>.</param>
    public int Rows(int width) => width <= 0 ? 0 : Values.Length / width;
}
