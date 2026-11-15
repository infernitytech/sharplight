using Sharplight.Llm.Embeddings;

namespace Sharplight.Llm.Scoring;

/// <summary>Options for <see cref="SpanScorer.Load"/>; <see cref="Tasks.TaskLoadOptions.MaxBatchTokens"/> bounds the prompt length.</summary>
public sealed class SpanScorerLoadOptions : PackedModelLoadOptions
{
    /// <summary>Most questions per prompt.</summary>
    public int MaxQuestions { get; set; } = 64;

    /// <summary>Most options per prompt (all questions together).</summary>
    public int MaxOptions { get; set; } = 512;
}
