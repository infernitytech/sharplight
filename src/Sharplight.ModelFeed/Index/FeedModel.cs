namespace Sharplight.ModelFeed.Index;

/// <summary>One model group of a <see cref="FeedModelListing"/>: shared model data and its variant packages.</summary>
public sealed record FeedModel
{
    /// <summary>Model group id (<see cref="PackageGroup.Id"/>: the catalog entry id).</summary>
    public required string Id { get; init; }

    /// <summary>Model name.</summary>
    public string? Name { get; init; }

    /// <summary>Model family.</summary>
    public string? Family { get; init; }

    /// <summary>Engine model kind (<c>text-generation</c>, ...).</summary>
    public string? Kind { get; init; }

    /// <summary>Architecture name.</summary>
    public string? Architecture { get; init; }

    /// <summary>Parameter count (of the recommended variant's package).</summary>
    public long? Parameters { get; init; }

    /// <summary>Maximum context length in tokens.</summary>
    public int? ContextLength { get; init; }

    /// <summary>Who made the model.</summary>
    public PackagePublisher? Publisher { get; init; }

    /// <summary>License.</summary>
    public PackageLicense? License { get; init; }

    /// <summary>Tags.</summary>
    public IReadOnlyList<string> Tags { get; init; } = [];

    /// <summary>Package id of the recommended variant, or <see langword="null"/> when none is marked.</summary>
    public string? Recommended { get; init; }

    /// <summary>The variants, smallest bits per weight first.</summary>
    public IReadOnlyList<FeedModelVariant> Variants { get; init; } = [];
}
