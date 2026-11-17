namespace Sharplight.ModelFeed.Resolution;

/// <summary>A package version chosen by resolution.</summary>
/// <param name="Candidate">The chosen version.</param>
/// <param name="RequiredBy">Ids of the packages that depend on it; empty for a requested package.</param>
/// <param name="Role">Dependency role (<c>voice-activity</c>, <c>draft</c>, ...), or <see langword="null"/> for a requested package.</param>
public sealed record ResolvedPackage(PackageCandidate Candidate, IReadOnlyList<string> RequiredBy, string? Role)
{
    /// <summary>Package id.</summary>
    public string Id => Candidate.Id;

    /// <summary>Chosen version.</summary>
    public string Version => Candidate.Version;

    /// <summary>Whether the caller requested it (not only a dependency).</summary>
    public bool Requested => RequiredBy.Count == 0;
}
