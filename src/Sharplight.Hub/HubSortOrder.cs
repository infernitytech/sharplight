namespace Sharplight.Hub;

/// <summary>How the Hub orders the repositories of a search (<see cref="HubSearchQuery.Sort"/>), best first.</summary>
public enum HubSortOrder
{
    /// <summary>Most downloaded in the last 30 days.</summary>
    Downloads,

    /// <summary>Most liked.</summary>
    Likes,

    /// <summary>The Hub's trending score (recent likes and downloads).</summary>
    Trending,

    /// <summary>Most recently created.</summary>
    Newest,
}
