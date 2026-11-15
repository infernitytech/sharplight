namespace Sharplight.Llm.Structured;

/// <summary>
/// A <see cref="TokenMaskCache"/> entry: the tokens a truncated stack accepts in every context (as mask words, or as a list
/// when few), and the token index ranges (into <see cref="TokenTrie.Tokens"/>) that must be checked against the real state.
/// </summary>
internal sealed record TokenMaskEntry(uint[]? Words, int[]? Tokens, (int Start, int End)[] Uncertain);
