using Sharplight.ModelFeed.Versioning;

namespace Sharplight.ModelFeed.Resolution;

/// <summary>Checks a package's requirements against the installing host before install and before load.</summary>
public static class PackageCompatibility
{
    /// <summary>Why the package cannot be installed here; empty when it can.</summary>
    /// <param name="requirements">The package's requirements, or <see langword="null"/>.</param>
    /// <param name="environment">The host.</param>
    /// <returns>Human-readable reasons.</returns>
    public static IReadOnlyList<string> Check(PackageRequirements? requirements, CompatibilityEnvironment environment)
    {
        ArgumentNullException.ThrowIfNull(environment);
        var reasons = new List<string>();
        if (requirements is null)
        {
            return reasons;
        }

        CheckVersion(requirements.Server, environment.HostVersion, "server", reasons);
        CheckVersion(requirements.Engine, environment.EngineVersion, "engine", reasons);
        if (requirements.PackageFormat is { } format && environment.MaxPackageFormat is { } maximum && format > maximum)
        {
            reasons.Add($"needs package format {format}; this engine reads up to {maximum}");
        }

        return reasons;
    }

    private static void CheckVersion(string? required, SemanticVersion? actual, string what, List<string> reasons)
    {
        if (string.IsNullOrWhiteSpace(required) || actual is null)
        {
            return;
        }

        if (!VersionRange.TryParse(required, out var range))
        {
            reasons.Add($"has an unreadable {what} requirement '{required}'");
            return;
        }

        if (!range.Satisfies(actual, includePrerelease: true))
        {
            reasons.Add($"needs {what} {required}; this is {actual}");
        }
    }
}
