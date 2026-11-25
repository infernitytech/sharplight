using Sharplight.ModelFeed.Index;

namespace Sharplight.ModelFeed.Tests;

/// <summary>Compares model listings by their serialized form (records with list members do not compare by value).</summary>
internal sealed class ListingComparer : IEqualityComparer<FeedModelListing>
{
    public static ListingComparer Instance { get; } = new();

    public bool Equals(FeedModelListing? x, FeedModelListing? y) => x is not null && y is not null && Text(x) == Text(y);

    public int GetHashCode(FeedModelListing obj) => Text(obj).GetHashCode(StringComparison.Ordinal);

    private static string Text(FeedModelListing listing) => System.Text.Encoding.UTF8.GetString(FeedJson.Write(listing, FeedJsonContext.Default.FeedModelListing));
}
