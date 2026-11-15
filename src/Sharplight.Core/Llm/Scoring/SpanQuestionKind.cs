namespace Sharplight.Llm.Scoring;

/// <summary>What a question of a span scoring prompt asks; heads that know question types (Clef) read it, others ignore it.</summary>
public enum SpanQuestionKind
{
    /// <summary>A yes/no (true/false) question (llama.cpp's "noul").</summary>
    Binary = 0,
    /// <summary>One of several options.</summary>
    Choice = 1,
    /// <summary>A level on a scale (the options are the levels).</summary>
    Score = 2,
}
