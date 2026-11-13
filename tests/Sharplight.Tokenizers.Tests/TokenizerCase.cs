namespace Sharplight.Tokenizers.Tests;

/// <summary>One golden encoding case of the Qwen3.5 fixture: the text, whether control tokens are parsed, and the expected ids.</summary>
internal sealed record TokenizerCase(string Text, bool ParseSpecial, int[] Ids);
