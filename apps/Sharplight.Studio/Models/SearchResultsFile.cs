namespace Sharplight.Studio.Models;

/// <summary>The result of a search as the page keeps it (<c>results.json</c> of the run).</summary>
internal sealed record SearchResultsFile
{
    /// <summary>The query.</summary>
    public string Query { get; init; } = "";

    /// <summary>The searched folder.</summary>
    public string Folder { get; init; } = "";

    /// <summary>The embedding model.</summary>
    public string Model { get; init; } = "";

    /// <summary>The search mode (<c>semantic</c>, <c>keyword</c>, <c>hybrid</c>).</summary>
    public string Mode { get; init; } = "semantic";

    /// <summary>The model's answer when the run was an Ask, else empty.</summary>
    public string Answer { get; init; } = "";

    /// <summary>The passages, best first.</summary>
    public List<SearchHitRecord> Hits { get; init; } = [];
}
