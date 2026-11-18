using Sharplight.ModelFeed.Index;
using Sharplight.ModelFeed.Resolution;
using Sharplight.Studio.Models;

namespace Sharplight.Studio.Services.FirstRun;

/// <summary>
/// Picks starter models from the feed's package metadata for a machine: a chat model sized to the GPU (or the CPU without one), then optionally
/// speech to text and text embeddings. Packages qualify by kind, task, memory figure and size, never by id; the preferences rank families.
/// Pure: the same listing, machine and preferences give the same recommendation.
/// </summary>
internal sealed class StarterRecommender
{
    private const long Mebibyte = 1L << 20;

    private readonly StarterPreferences _preferences;

    /// <param name="preferences">The ranking and limits.</param>
    public StarterRecommender(StarterPreferences preferences)
    {
        _preferences = preferences;
    }

    /// <summary>Recommends starter models.</summary>
    /// <param name="packages">The feed listing (empty offline without a saved copy).</param>
    /// <param name="machine">The machine.</param>
    /// <param name="environment">What package requirements are checked against.</param>
    public StarterRecommendation Recommend(IReadOnlyList<FeedPackageSummary> packages, MachineProfile machine, CompatibilityEnvironment environment)
    {
        var byId = packages.ToDictionary(package => package.Id, StringComparer.Ordinal);
        var usable = packages.Where(package => IsUsable(package, environment)).ToList();
        string prompt = _preferences.ExamplePrompts.FirstOrDefault() ?? "";
        long cpuBudget = CpuBudget(machine);
        long gpuBudget = machine.HasGpu ? (long)(machine.GpuMemoryBytes * _preferences.GpuMemoryFraction) : 0;
        long budget = Math.Max(gpuBudget, cpuBudget);
        bool cpuOnly = gpuBudget < cpuBudget;
        if (usable.Count == 0)
        {
            return new StarterRecommendation
            {
                Outcome = StarterOutcome.NoPackages,
                Message = "The model feed lists no packages this version of Studio can install.",
                CpuOnly = cpuOnly,
                MemoryBudgetBytes = budget,
                ExamplePrompt = prompt,
            };
        }

        long diskLeft = machine.FreeDiskBytes is { } free ? free - _preferences.DiskReserveMiB * Mebibyte : long.MaxValue;
        var counted = new HashSet<string>(StringComparer.Ordinal);
        var items = new List<StarterItem>();
        long remainingGpu = gpuBudget;
        foreach (var role in _preferences.Roles)
        {
            bool isChat = role.Role == StarterRole.Chat;
            if (!isChat && items.Count == 0)
            {
                break;
            }

            long limit = isChat ? budget : Math.Max(remainingGpu, cpuBudget);
            var pick = Choose(role, usable, byId, limit, diskLeft, counted);
            if (pick is null)
            {
                continue;
            }

            long memory = MemoryOf(pick.Value.Package);
            bool onGpu = machine.HasGpu && memory <= (isChat ? gpuBudget : remainingGpu);
            if (onGpu)
            {
                remainingGpu -= memory;
            }

            foreach (string id in pick.Value.NewIds)
            {
                counted.Add(id);
            }

            diskLeft = diskLeft == long.MaxValue ? diskLeft : diskLeft - pick.Value.DownloadBytes;
            items.Add(new StarterItem(role.Role, pick.Value.Package, pick.Value.DownloadBytes, memory, onGpu, role.Preselected || isChat, EstimateTime(pick.Value.DownloadBytes)));
        }

        if (items.Count == 0 || items[0].Role != StarterRole.Chat)
        {
            return new StarterRecommendation
            {
                Outcome = StarterOutcome.NothingFits,
                Message = Explain(machine, budget, usable),
                CpuOnly = cpuOnly,
                MemoryBudgetBytes = budget,
                ExamplePrompt = prompt,
            };
        }

        return new StarterRecommendation
        {
            Outcome = StarterOutcome.Recommended,
            Items = items,
            CpuOnly = cpuOnly,
            MemoryBudgetBytes = budget,
            ExamplePrompt = prompt,
        };
    }

    /// <summary>The time to download <paramref name="bytes"/> at the assumed connection speed.</summary>
    /// <param name="bytes">The download size.</param>
    public TimeSpan EstimateTime(long bytes)
    {
        double bytesPerSecond = Math.Max(0.1, _preferences.AssumedDownloadMbps) * 1_000_000 / 8;
        return TimeSpan.FromSeconds(Math.Max(0, bytes) / bytesPerSecond);
    }

    /// <summary>The memory a package needs on its device (the feed's figure for one slot, else its size).</summary>
    /// <param name="package">The package.</param>
    public static long MemoryOf(FeedPackageSummary package)
    {
        var memory = package.Requirements?.Memory ?? [];
        var gpu = memory.FirstOrDefault(entry => string.Equals(entry.Device, "gpu", StringComparison.OrdinalIgnoreCase)) ?? memory.FirstOrDefault();
        return gpu is not null ? gpu.TotalFor(1) : package.Size;
    }

    private long CpuBudget(MachineProfile machine)
    {
        long cap = _preferences.CpuMaxMiB * Mebibyte;
        return machine.SystemMemoryBytes <= 0 ? cap / 2 : Math.Min(cap, (long)(machine.SystemMemoryBytes * _preferences.CpuMemoryFraction));
    }

    private static bool IsUsable(FeedPackageSummary package, CompatibilityEnvironment environment)
        => package.State?.Yanked is null && package.State?.Deprecated is null && PackageCompatibility.Check(package.Requirements, environment).Count == 0;

    private (FeedPackageSummary Package, long DownloadBytes, IReadOnlyList<string> NewIds)? Choose(StarterRolePreference role, List<FeedPackageSummary> usable,
        Dictionary<string, FeedPackageSummary> byId, long memoryLimit, long diskLeft, HashSet<string> counted)
    {
        var candidates = usable
            .Where(package => string.Equals(package.Model?.Kind, role.Kind, StringComparison.OrdinalIgnoreCase)
                && (role.Task is null || package.Model?.Tasks.Contains(role.Task, StringComparer.OrdinalIgnoreCase) == true))
            .OrderBy(package => Rank(role, package))
            .ThenByDescending(MemoryOf)
            .ThenBy(package => package.Id, StringComparer.Ordinal);
        long maxDownload = (role.MaxDownloadMiB > 0 ? role.MaxDownloadMiB : role.Role == StarterRole.Chat ? _preferences.MaxDownloadMiB : 0) * Mebibyte;
        foreach (var package in candidates)
        {
            if (MemoryOf(package) > memoryLimit || !TryCollect(package, byId, counted, out long download, out var newIds))
            {
                continue;
            }

            if (download > diskLeft || maxDownload > 0 && download > maxDownload)
            {
                continue;
            }

            return (package, download, newIds);
        }

        return null;
    }

    private static bool TryCollect(FeedPackageSummary package, Dictionary<string, FeedPackageSummary> byId, HashSet<string> counted, out long bytes, out IReadOnlyList<string> newIds)
    {
        var ids = new List<string>();
        bytes = 0;
        newIds = ids;
        var pending = new Stack<FeedPackageSummary>();
        pending.Push(package);
        while (pending.Count > 0)
        {
            var current = pending.Pop();
            if (counted.Contains(current.Id) || ids.Contains(current.Id))
            {
                continue;
            }

            ids.Add(current.Id);
            bytes += current.Size;
            foreach (var dependency in current.Dependencies.Where(dependency => dependency.IsRequired))
            {
                if (!byId.TryGetValue(dependency.Id, out var required))
                {
                    return false;
                }

                pending.Push(required);
            }
        }

        return true;
    }

    private static int Rank(StarterRolePreference role, FeedPackageSummary package)
    {
        string text = string.Join(' ', package.Display?.Family, package.Display?.Name, package.Model?.Architecture, package.Id);
        for (int index = 0; index < role.Prefer.Count; index++)
        {
            if (text.Contains(role.Prefer[index], StringComparison.OrdinalIgnoreCase))
            {
                return index;
            }
        }

        return role.Prefer.Count;
    }

    private string Explain(MachineProfile machine, long budget, List<FeedPackageSummary> usable)
    {
        var chat = _preferences.Roles.FirstOrDefault(role => role.Role == StarterRole.Chat);
        bool fitsMemory = chat is not null && usable.Any(package => string.Equals(package.Model?.Kind, chat.Kind, StringComparison.OrdinalIgnoreCase) && MemoryOf(package) <= budget);
        if (fitsMemory && machine.FreeDiskBytes is { } free)
        {
            return $"There is not enough free disk space for a model ({ByteFormat.Format(free)} free where models are stored). Free some space or choose another storage location in the settings.";
        }

        return $"None of the starter models fits this machine's memory (about {ByteFormat.Format(budget)} available for a model). You can still browse the feed and the Hub for something smaller.";
    }
}
