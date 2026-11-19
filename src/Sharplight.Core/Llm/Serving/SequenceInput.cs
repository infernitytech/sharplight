namespace Sharplight.Llm.Serving;

/// <summary>Rows one sequence contributes to a <see cref="IBatchDecoder.Step"/>.</summary>
/// <param name="Slot">The sequence slot (0 .. <see cref="IBatchDecoder.SlotCount"/> - 1) whose context the rows extend.</param>
/// <param name="Tokens">
/// Token ids appended to the slot's context (prompt chunk, generated token, or a token plus draft tokens). Rows filled from
/// <see cref="Embeddings"/> keep their token id (the placeholder), which only identifies them for prefix caching.
/// </param>
/// <param name="WantLogits">Whether logits are needed (false for non-final prompt chunks).</param>
/// <param name="LogitsRows">
/// How many trailing rows of <paramref name="Tokens"/> return logits when <paramref name="WantLogits"/> (1 = the last row only;
/// more for speculative verification, at most <see cref="IBatchDecoder.MaxLogitsRows"/> summed over a step).
/// </param>
/// <param name="Speculative">
/// The rows after the first are draft tokens: right after the step, <see cref="IBatchDecoder.Truncate"/> may drop any of the
/// input's rows (at most <see cref="IBatchDecoder.MaxSpeculativeRows"/> rows), also on models with recurrent state.
/// </param>
public readonly record struct SequenceInput(int Slot, ReadOnlyMemory<int> Tokens, bool WantLogits = true, int LogitsRows = 1, bool Speculative = false)
{
    /// <summary>Rows of logits this input produces: <see cref="LogitsRows"/> clamped to [1, token count], or 0 without logits.</summary>
    public int OutputRows => WantLogits ? Math.Clamp(LogitsRows, 1, Math.Max(1, Tokens.Length)) : 0;

    /// <summary>
    /// Rows whose input is an embedding from a vision or audio encoder instead of their token's (non-overlapping, inside the input);
    /// null or empty for token rows only. Needs a decoder with <see cref="IBatchDecoder.EmbeddingWidth"/> &gt; 0.
    /// </summary>
    public IReadOnlyList<EmbeddingRun>? Embeddings { get; init; }

    /// <summary>
    /// Rope position ids of the rows, [rows, <see cref="IBatchDecoder.PositionAxes"/>] row-major (M-RoPE prompts with image or audio spans,
    /// whose positions are not their cache positions). Empty: every row's position is its cache position plus <see cref="PositionOffset"/>
    /// on the text axes (0 on the others).
    /// </summary>
    public ReadOnlyMemory<int> Positions { get; init; }

    /// <summary>
    /// Added to the cache position of each row to give its rope position when <see cref="Positions"/> is empty: the rows after an M-RoPE
    /// image span continue at a position below their cache position (vLLM's <c>mrope_position_delta</c>). 0 for plain text.
    /// </summary>
    public int PositionOffset { get; init; }

    /// <summary>
    /// Per row, whether it goes through the language model's media weight set (<see cref="Models.InputRole.MediaWeightMask"/>, CogVLM); empty
    /// for the default, which is the <see cref="Embeddings"/> rows' <see cref="EmbeddingRun.MediaWeights"/> and no other row.
    /// </summary>
    public ReadOnlyMemory<bool> MediaWeights { get; init; }

    /// <summary>Whether the input carries anything besides token ids (embeddings, explicit positions, an offset or media weights).</summary>
    public bool HasMultimodalRows => Embeddings is { Count: > 0 } || !Positions.IsEmpty || PositionOffset != 0 || !MediaWeights.IsEmpty;
}
