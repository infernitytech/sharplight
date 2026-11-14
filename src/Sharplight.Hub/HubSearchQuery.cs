namespace Sharplight.Hub;

/// <summary>A model search on the Hub, sorted by downloads unless <see cref="Sort"/> says otherwise.</summary>
public sealed class HubSearchQuery
{
    /// <summary>Default <see cref="Limit"/>.</summary>
    public const int DefaultLimit = 20;

    /// <summary>Text matched against repository ids; empty lists every repository (in <see cref="Sort"/> order).</summary>
    public string Text { get; init; } = "";

    /// <summary>The order of the results.</summary>
    public HubSortOrder Sort { get; init; } = HubSortOrder.Downloads;

    /// <summary>Only repositories tagged with this format, or <see langword="null"/> for any.</summary>
    public ModelFormat? Format { get; init; }

    /// <summary>Maximum number of results.</summary>
    public int Limit { get; init; } = DefaultLimit;
}
