namespace Sharplight.ModelFeed.Resolution;

/// <summary>The outcome of <see cref="DependencyResolver.ResolveAsync"/>: every package to install, or the problems that prevent it.</summary>
/// <param name="Packages">Chosen versions, requested packages first, then dependencies by id.</param>
/// <param name="Problems">Why resolution failed; empty on success.</param>
public sealed record ResolutionResult(IReadOnlyList<ResolvedPackage> Packages, IReadOnlyList<string> Problems)
{
    /// <summary>Whether every request and required dependency was resolved.</summary>
    public bool Succeeded => Problems.Count == 0;
}
