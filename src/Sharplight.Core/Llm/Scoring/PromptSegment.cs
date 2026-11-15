namespace Sharplight.Llm.Scoring;

/// <summary>
/// One piece of a span scoring prompt. Every segment is tokenized on its own (no special tokens), so a span starts and ends at a segment
/// boundary; the model's prompt format decides where the boundaries are (e.g. the start of an option is a segment of its own).
/// </summary>
/// <param name="Text">The text of the segment.</param>
/// <param name="Role">Whether the head reads the segment as a question or an option span.</param>
/// <param name="Kind">The question kind (only for <see cref="PromptSegmentRole.Question"/>).</param>
public sealed record PromptSegment(string Text, PromptSegmentRole Role = PromptSegmentRole.Text, SpanQuestionKind Kind = SpanQuestionKind.Choice);
