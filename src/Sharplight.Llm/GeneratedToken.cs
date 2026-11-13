namespace Sharplight.Llm;

/// <summary>One streamed piece of generated text.</summary>
/// <param name="Id">The sampled token id, or -1 for the final text flushed from the incremental UTF-8 decoder.</param>
/// <param name="Text">Newly decodable text (may be empty while a multi-byte character is incomplete).</param>
public readonly record struct GeneratedToken(int Id, string Text);
