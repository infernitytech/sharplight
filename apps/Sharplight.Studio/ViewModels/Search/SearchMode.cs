namespace Sharplight.Studio.ViewModels.Search;

/// <summary>How a search ranks the passages.</summary>
internal static class SearchMode
{
    /// <summary>Embedding similarity only.</summary>
    public const string Semantic = "semantic";

    /// <summary>BM25 keyword matching only.</summary>
    public const string Keyword = "keyword";

    /// <summary>BM25 and embeddings combined by reciprocal rank fusion.</summary>
    public const string Hybrid = "hybrid";
}
