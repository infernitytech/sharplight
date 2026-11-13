namespace Sharplight.Llm.Serving;

/// <summary>Rows one sequence contributes to a <see cref="IBatchDecoder.Step"/>.</summary>
/// <param name="Slot">The sequence slot (0 .. <see cref="IBatchDecoder.SlotCount"/> - 1) whose context the rows extend.</param>
/// <param name="Tokens">Token ids appended to the slot's context (prompt chunk, generated token, or a token plus draft tokens).</param>
/// <param name="WantLogits">Whether logits are needed (false for non-final prompt chunks).</param>
/// <param name="LogitsRows">
/// How many trailing rows of <paramref name="Tokens"/> return logits when <paramref name="WantLogits"/> (1 = the last row only;
/// more for speculative verification, at most <see cref="IBatchDecoder.MaxLogitsRows"/> summed over a step).
/// </param>
public readonly record struct SequenceInput(int Slot, ReadOnlyMemory<int> Tokens, bool WantLogits = true, int LogitsRows = 1)
{
    /// <summary>Rows of logits this input produces: <see cref="LogitsRows"/> clamped to [1, token count], or 0 without logits.</summary>
    public int OutputRows => WantLogits ? Math.Clamp(LogitsRows, 1, Math.Max(1, Tokens.Length)) : 0;
}
