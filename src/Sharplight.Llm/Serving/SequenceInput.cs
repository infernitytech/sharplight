namespace Sharplight.Llm.Serving;

/// <summary>Rows one sequence contributes to a <see cref="IBatchDecoder.Step"/>.</summary>
/// <param name="Slot">The sequence slot (0 .. <see cref="IBatchDecoder.SlotCount"/> - 1) whose context the rows extend.</param>
/// <param name="Tokens">Token ids appended to the slot's context (prompt chunk or generated token).</param>
/// <param name="WantLogits">Whether the logits of the last row are needed (false for non-final prompt chunks).</param>
public readonly record struct SequenceInput(int Slot, ReadOnlyMemory<int> Tokens, bool WantLogits = true);
