namespace Sharplight.ModelFeed.Resolution;

/// <summary>Options of <see cref="DependencyResolver"/>.</summary>
public sealed record ResolutionOptions
{
    /// <summary>The installing host; incompatible versions are skipped with their reasons recorded.</summary>
    public CompatibilityEnvironment Environment { get; init; } = new();

    /// <summary>Whether prerelease versions may be chosen when the range does not name one.</summary>
    public bool IncludePrerelease { get; init; }

    /// <summary>Optional dependencies to include (by id); required ones are always included.</summary>
    public IReadOnlySet<string> IncludeOptional { get; init; } = new HashSet<string>(StringComparer.Ordinal);

    /// <summary>Versions to keep when they satisfy every range (installed or locked versions), by id.</summary>
    public IReadOnlyDictionary<string, string> Preferred { get; init; } = new Dictionary<string, string>(StringComparer.Ordinal);
}
