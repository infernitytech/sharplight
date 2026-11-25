namespace Sharplight.ModelFeed;

/// <summary>Quality tiers of <see cref="PackageGroup.Quality"/>, best first: how a variant compares with the group's reference precision.</summary>
public static class PackageQualityTiers
{
    /// <summary>The reference precision itself (the published weights, BF16 or the native release format).</summary>
    public const string Reference = "reference";

    /// <summary>Indistinguishable from the reference in practice (8-bit, FP8).</summary>
    public const string NearLossless = "near-lossless";

    /// <summary>Small loss, the usual choice (5-6 bit).</summary>
    public const string High = "high";

    /// <summary>Noticeable but moderate loss (4 bit).</summary>
    public const string Good = "good";

    /// <summary>Clear loss; for memory-constrained hosts (3 bit).</summary>
    public const string Reduced = "reduced";

    /// <summary>Large loss; runs where nothing else fits (1-2 bit).</summary>
    public const string Low = "low";

    /// <summary>Every tier, best first.</summary>
    public static IReadOnlyList<string> All { get; } = [Reference, NearLossless, High, Good, Reduced, Low];
}
