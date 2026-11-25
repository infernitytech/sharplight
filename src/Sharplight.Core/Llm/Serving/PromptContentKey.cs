namespace Sharplight.Llm.Serving;

/// <summary>
/// Non-text content that occupies prompt positions, such as the placeholder tokens of an image or an audio clip whose embeddings
/// replace them: the tokens alone do not identify what the model sees there, so the prefix cache mixes <paramref name="Hash"/>
/// (a hash of the content, e.g. of the encoded image bytes) into the hash of every KV block overlapping the span.
/// </summary>
/// <param name="Start">First prompt position of the content.</param>
/// <param name="Length">Prompt positions it occupies.</param>
/// <param name="Hash">Hash of the content (collision resistant, e.g. 128 bits of SHA-256).</param>
public readonly record struct PromptContentKey(int Start, int Length, UInt128 Hash);
