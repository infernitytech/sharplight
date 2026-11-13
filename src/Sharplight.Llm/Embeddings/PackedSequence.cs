namespace Sharplight.Llm.Embeddings;

/// <summary>One input sequence of a packed-sequence graph: token ids and (for encoders with token types) their type ids.</summary>
public readonly record struct PackedSequence(int[] Tokens, int[]? TokenTypes = null);
