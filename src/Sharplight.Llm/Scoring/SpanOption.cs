namespace Sharplight.Llm.Scoring;

/// <summary>An option span of a tokenized span scoring prompt.</summary>
/// <param name="Start">Index of its first token.</param>
/// <param name="Length">Its token count (at least 1).</param>
/// <param name="Question">Index of the question it belongs to.</param>
public readonly record struct SpanOption(int Start, int Length, int Question);
