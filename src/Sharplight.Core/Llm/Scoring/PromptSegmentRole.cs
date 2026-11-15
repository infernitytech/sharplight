namespace Sharplight.Llm.Scoring;

/// <summary>What the span scoring head makes of a <see cref="PromptSegment"/>.</summary>
public enum PromptSegmentRole
{
    /// <summary>Plain prompt text, not read as a span.</summary>
    Text,
    /// <summary>The text of a question: options that follow belong to it.</summary>
    Question,
    /// <summary>The text of an option of the last question before it.</summary>
    Option,
}
