namespace Sharplight.Llm.Scoring;

/// <summary>A question span of a tokenized span scoring prompt.</summary>
/// <param name="Start">Index of its first token.</param>
/// <param name="Length">Its token count (at least 1).</param>
/// <param name="Kind">What the question asks.</param>
public readonly record struct SpanQuestion(int Start, int Length, SpanQuestionKind Kind = SpanQuestionKind.Choice);
