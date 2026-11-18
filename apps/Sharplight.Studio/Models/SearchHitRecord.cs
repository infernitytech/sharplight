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

    /// <summary>Cosine similarity to the query.</summary>
    public double Score { get; init; }
}
