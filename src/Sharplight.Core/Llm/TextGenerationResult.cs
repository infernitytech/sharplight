namespace Sharplight.Llm;

/// <summary>The result of <see cref="ITextGenerator.GenerateAsync"/>.</summary>
/// <param name="Text">The generated text.</param>
/// <param name="Tokens">The generated pieces in order (the last may be the flushed tail of the UTF-8 decoder, id -1).</param>
/// <param name="Stats">Token counts and timings, or <see langword="null"/> when the implementation does not measure them.</param>
public sealed record TextGenerationResult(string Text, IReadOnlyList<GeneratedToken> Tokens, GenerationStats? Stats);
