namespace Sharplight.Studio.Models;

/// <summary>A passage a search found.</summary>
internal sealed record SearchHitRecord
{
    /// <summary>The document's path.</summary>
    public string File { get; init; } = "";

    /// <summary>Where the passage starts in the document's text.</summary>
    public int Start { get; init; }

    /// <summary>The passage.</summary>
    public string Text { get; init; } = "";

    /// <summary>The page, slide or sheet number (1-based), or 0.</summary>
    public int Page { get; init; }

    /// <summary>The line of the file the passage starts at (1-based), or 0.</summary>
    public int Line { get; init; }

    /// <summary>The heading the passage stands under, or empty.</summary>
    public string Heading { get; init; } = "";

    /// <summary>The score: cosine similarity for a semantic search, the share of the best fused score for a hybrid one, the share of the best BM25 score for a keyword one, the reranker's own score when reranked.</summary>
    public double Score { get; init; }
}
