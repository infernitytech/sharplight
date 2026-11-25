namespace Sharplight.ModelFeed.Index;

/// <summary>One variant package of a <see cref="FeedModel"/>: its place in the group and what a compatibility check needs.</summary>
public sealed record FeedModelVariant
{
    /// <summary>Package id (<c>&lt;group&gt;.&lt;variant&gt;</c>).</summary>
    public required string Package { get; init; }

    /// <summary>Latest listed version in the channel.</summary>
    public required string Version { get; init; }

    /// <summary>Variant id within the group.</summary>
    public string? Variant { get; init; }

    /// <summary>Variant label.</summary>
    public string? Label { get; init; }

    /// <summary>Upstream format.</summary>
    public string? Format { get; init; }

    /// <summary>Quantization.</summary>
    public string? Quantization { get; init; }

    /// <summary>Average stored bits per weight.</summary>
    public double? BitsPerWeight { get; init; }

    /// <summary>The recommended variant of the model.</summary>
    public bool Recommended { get; init; }

    /// <summary>Quality tier relative to <see cref="Reference"/> (<see cref="PackageQualityTiers"/>).</summary>
    public string? Quality { get; init; }

    /// <summary>Quality note.</summary>
    public string? QualityNote { get; init; }

    /// <summary>Reference precision.</summary>
    public string? Reference { get; init; }

    /// <summary>Package file bytes (download size).</summary>
    public long Size { get; init; }

    /// <summary>Verification level of the variant (<c>verified</c>, <c>partial</c>).</summary>
    public string? Verification { get; init; }

    /// <summary>Requirements (memory per device kind, kernels, device features, disk, versions).</summary>
    public PackageRequirements? Requirements { get; init; }

    /// <summary>Current state of the latest version.</summary>
    public PackageState? State { get; init; }

    /// <summary>The variant entry of a package summary.</summary>
    /// <param name="summary">The summary.</param>
    /// <returns>The variant.</returns>
    public static FeedModelVariant From(FeedPackageSummary summary)
    {
        ArgumentNullException.ThrowIfNull(summary);
        var group = summary.Group;
        return new FeedModelVariant
        {
            Package = summary.Id,
            Version = summary.LatestVersion,
            Variant = group?.Variant,
            Label = group?.Label ?? summary.Display?.Variant,
            Format = group?.Format,
            Quantization = group?.Quantization ?? summary.Model?.Quantization,
            BitsPerWeight = group?.BitsPerWeight,
            Recommended = group?.Recommended ?? false,
            Quality = group?.Quality,
            QualityNote = group?.QualityNote,
            Reference = group?.Reference,
            Size = summary.Size,
            Verification = summary.Verification?.Level,
            Requirements = summary.Requirements,
            State = summary.State,
        };
    }
}
