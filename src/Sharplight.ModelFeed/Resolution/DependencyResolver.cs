using Sharplight.ModelFeed.Versioning;

namespace Sharplight.ModelFeed.Resolution;

/// <summary>
/// Resolves requested packages and their dependencies: for each id the highest version (or the preferred one) that satisfies every
/// range on it, is compatible with the host, and is not yanked; repeated until the choice is stable. Conflicts are reported and
/// nothing is chosen partially. Thread safe (stateless).
/// </summary>
public sealed class DependencyResolver
{
    private const int MaxRounds = 64;

    /// <summary>Resolves <paramref name="requests"/>.</summary>
    /// <param name="requests">The packages wanted.</param>
    /// <param name="candidates">Available versions of a package id (empty when unknown).</param>
    /// <param name="options">Host, prerelease and optional dependency choices.</param>
    /// <param name="cancellationToken">Cancellation.</param>
    /// <returns>The result.</returns>
    public async Task<ResolutionResult> ResolveAsync(
        IReadOnlyList<PackageRequest> requests,
        Func<string, CancellationToken, Task<IReadOnlyList<PackageCandidate>>> candidates,
        ResolutionOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(requests);
        ArgumentNullException.ThrowIfNull(candidates);
        options ??= new ResolutionOptions();
        var available = new Dictionary<string, IReadOnlyList<PackageCandidate>>(StringComparer.Ordinal);
        var chosen = new Dictionary<string, PackageCandidate>(StringComparer.Ordinal);
        for (var round = 0; round < MaxRounds; round++)
        {
            var constraints = Constraints(requests, chosen, options);
            var problems = new List<string>();
            var next = new Dictionary<string, PackageCandidate>(StringComparer.Ordinal);
            foreach (var (id, ranges) in constraints)
            {
                if (!available.TryGetValue(id, out var versions))
                {
                    versions = await candidates(id, cancellationToken).ConfigureAwait(false);
                    available[id] = versions;
                }

                var pick = Choose(id, versions, ranges, options, problems);
                if (pick is not null)
                {
                    next[id] = pick;
                }
            }

            if (problems.Count > 0)
            {
                return new ResolutionResult([], problems);
            }

            if (Same(chosen, next))
            {
                return new ResolutionResult(Describe(requests, next, constraints), []);
            }

            chosen = next;
        }

        return new ResolutionResult([], ["The dependencies do not settle on one set of versions (cycle of changing choices)."]);
    }

    private static Dictionary<string, List<Constraint>> Constraints(IReadOnlyList<PackageRequest> requests, Dictionary<string, PackageCandidate> chosen, ResolutionOptions options)
    {
        var constraints = new Dictionary<string, List<Constraint>>(StringComparer.Ordinal);
        foreach (var request in requests)
        {
            Add(constraints, request.Id, new Constraint(Parse(request.Range, request.Id), null, null));
        }

        foreach (var candidate in chosen.Values)
        {
            foreach (var dependency in candidate.Dependencies)
            {
                if (dependency.IsRequired || options.IncludeOptional.Contains(dependency.Id))
                {
                    Add(constraints, dependency.Id, new Constraint(Parse(dependency.Range, dependency.Id), candidate.Id, dependency.Role));
                }
            }
        }

        return constraints;
    }

    private static PackageCandidate? Choose(string id, IReadOnlyList<PackageCandidate> versions, List<Constraint> ranges, ResolutionOptions options, List<string> problems)
    {
        var reasons = new List<string>();
        var eligible = new List<(SemanticVersion Version, PackageCandidate Candidate)>();
        foreach (var candidate in versions)
        {
            if (!SemanticVersion.TryParse(candidate.Version, out var version))
            {
                continue;
            }

            if (candidate.State?.Yanked is not null)
            {
                reasons.Add($"{version} is yanked");
                continue;
            }

            if (!ranges.All(constraint => constraint.Range.Satisfies(version, options.IncludePrerelease)))
            {
                continue;
            }

            var pinned = ranges.Any(constraint => constraint.Range.Minimum == version && constraint.Range.Maximum == version);
            if (candidate.State is { Listed: false } && !pinned)
            {
                continue;
            }

            var incompatible = PackageCompatibility.Check(candidate.Requirements, options.Environment);
            if (incompatible.Count > 0)
            {
                reasons.Add($"{version} {string.Join("; ", incompatible)}");
                continue;
            }

            eligible.Add((version, candidate));
        }

        if (eligible.Count == 0)
        {
            var wanted = string.Join(", ", ranges.Select(constraint => constraint.From is null ? $"{constraint.Range} (requested)" : $"{constraint.Range} (by {constraint.From})"));
            var why = versions.Count == 0 ? "the feed has no such package" : reasons.Count > 0 ? string.Join("; ", reasons) : "no version is in the range";
            problems.Add($"{id}: nothing satisfies {wanted}: {why}.");
            return null;
        }

        if (options.Preferred.TryGetValue(id, out var preferredText) && SemanticVersion.TryParse(preferredText, out var preferred))
        {
            var kept = eligible.FirstOrDefault(entry => entry.Version == preferred);
            if (kept.Candidate is not null)
            {
                return kept.Candidate;
            }
        }

        return eligible.MaxBy(entry => entry.Version)!.Candidate;
    }

    private static List<ResolvedPackage> Describe(IReadOnlyList<PackageRequest> requests, Dictionary<string, PackageCandidate> chosen, Dictionary<string, List<Constraint>> constraints)
    {
        var requested = requests.Select(request => request.Id).ToHashSet(StringComparer.Ordinal);
        return [.. chosen.Values
            .OrderBy(candidate => requested.Contains(candidate.Id) ? 0 : 1).ThenBy(candidate => candidate.Id, StringComparer.Ordinal)
            .Select(candidate =>
            {
                var from = constraints[candidate.Id];
                IReadOnlyList<string> requiredBy = requested.Contains(candidate.Id) ? [] : [.. from.Where(constraint => constraint.From is not null).Select(constraint => constraint.From!).Distinct(StringComparer.Ordinal)];
                return new ResolvedPackage(candidate, requiredBy, from.Select(constraint => constraint.Role).FirstOrDefault(role => role is not null));
            })];
    }

    private static bool Same(Dictionary<string, PackageCandidate> left, Dictionary<string, PackageCandidate> right) =>
        left.Count == right.Count && left.All(entry => right.TryGetValue(entry.Key, out var other) && other.Version == entry.Value.Version);

    private static void Add(Dictionary<string, List<Constraint>> constraints, string id, Constraint constraint)
    {
        if (!constraints.TryGetValue(id, out var list))
        {
            list = [];
            constraints[id] = list;
        }

        list.Add(constraint);
    }

    private static VersionRange Parse(string? range, string id)
    {
        return VersionRange.TryParse(range, out var parsed) ? parsed : throw new FeedFormatException($"The version range '{range}' for {id} is not valid.");
    }
}
