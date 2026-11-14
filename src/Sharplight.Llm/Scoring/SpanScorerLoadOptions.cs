using Sharplight.Llm.Embeddings;

namespace Sharplight.Llm.Scoring;

/// <summary>Options for <see cref="SpanScorer.Load"/>; <see cref="PackedModelLoadOptions.MaxBatchTokens"/> bounds the prompt length.</summary>
public sealed class SpanScorerLoadOptions : PackedModelLoadOptions
{
    /// <summary>Most questions per prompt.</summary>
    public int MaxQuestions { get; init; } = 64;
    /// <summary>Most options per prompt (all questions together).</summary>
    public int MaxOptions { get; init; } = 512;
}
