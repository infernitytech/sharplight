namespace Sharplight.ModelFeed;

/// <summary>How a model was verified (from the verified catalog) and that the built package was checked.</summary>
public sealed record PackageVerification
{
    /// <summary>Catalog level (<c>verified</c>, <c>partial</c>).</summary>
    public string? Level { get; init; }

    /// <summary>Reference implementation compared against.</summary>
    public string? Reference { get; init; }

    /// <summary>Result summary.</summary>
    public string? Summary { get; init; }

    /// <summary>Execution providers verified.</summary>
    public IReadOnlyList<string> Providers { get; init; } = [];

    /// <summary>Verification date.</summary>
    public string? Date { get; init; }

    /// <summary>Check of the built package.</summary>
    public PackageCheck? PackageCheck { get; init; }
}
