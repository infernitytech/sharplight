namespace Sharplight.Hosting.Fit;

/// <summary>
/// Decides whether a footprint fits the chosen devices the way the engine's model host admits a load: each device must have its
/// share of the footprint free beyond the reserve; the host first unloads idle, unpinned models holding memory there, and when the
/// first device is still short and CPU offload is on, it caps that device and the layers that do not fit run on the CPU.
/// </summary>
public static class FitPlanner
{
    /// <summary>The smallest weight a device gets when a model is split (so a device that is nearly full still gets a few layers).</summary>
    public const long MinSplitWeight = 256L << 20;

    /// <summary>The share weight of a device when a model is split over several: its free memory beyond the reserve.</summary>
    /// <param name="freeBytes">Free memory of the device.</param>
    /// <param name="reserveBytes">The reserve.</param>
    public static double SplitWeight(long freeBytes, long reserveBytes) => Math.Max(MinSplitWeight, freeBytes - reserveBytes);

    /// <summary>Plans the load.</summary>
    /// <param name="query">Footprint, devices, reserve, resident models and the offload setting.</param>
    public static FitReport Plan(FitQuery query)
    {
        var targets = query.Targets;
        var footprint = query.Footprint;
        if (targets.Count == 0)
        {
            return new FitReport { Verdict = FitVerdict.DoesNotFit, Footprint = footprint, Devices = [], CpuOffload = query.CpuOffload };
        }

        var needs = Shares(footprint.Total, targets, query.ReserveBytes);
        var none = new Dictionary<string, long>(StringComparer.Ordinal);
        var plain = Fits(targets, needs, none, query.ReserveBytes);
        if (plain.All(fit => fit.Fits))
        {
            return new FitReport { Verdict = FitVerdict.Fits, Footprint = footprint, Devices = plain, CpuOffload = query.CpuOffload };
        }

        // Unload resident models holding memory on a device that is short: idle unpinned ones first (the host does that by itself),
        // least recently used first.
        var ids = targets.Select(target => target.Id).ToHashSet(StringComparer.Ordinal);
        var candidates = query.Residents
            .Where(model => model.Bytes > 0 && model.DeviceIds.Any(ids.Contains))
            .OrderBy(model => model.CanEvict ? 0 : 1)
            .ThenBy(model => model.LastUsed ?? DateTimeOffset.MinValue)
            .ToList();
        var freed = new Dictionary<string, long>(StringComparer.Ordinal);
        var unload = new List<ResidentUsage>();
        var current = plain;
        foreach (var model in candidates)
        {
            var shortIds = current.Where(fit => !fit.Fits).Select(fit => fit.Device.Id).ToHashSet(StringComparer.Ordinal);
            if (shortIds.Count == 0)
            {
                break;
            }

            if (!model.DeviceIds.Any(shortIds.Contains))
            {
                continue;
            }

            unload.Add(model);
            Free(freed, model);
            current = Fits(targets, needs, freed, query.ReserveBytes);
        }

        if (current.All(fit => fit.Fits))
        {
            return new FitReport
            {
                Verdict = FitVerdict.FitsAfterUnloading,
                Footprint = footprint,
                Devices = current,
                Unload = [.. unload.Select(model => model.Id)],
                UnloadIsAutomatic = unload.All(model => model.CanEvict),
                CpuOffload = query.CpuOffload,
            };
        }

        return PlanOffload(query, needs, candidates);
    }

    /// <summary>
    /// The host unloads the idle models on the first device, then caps it at what is free: the layers beyond that (its share of
    /// weights and state minus the room left after activations) go to the CPU.
    /// </summary>
    private static FitReport PlanOffload(FitQuery query, long[] needs, List<ResidentUsage> candidates)
    {
        var targets = query.Targets;
        var footprint = query.Footprint;
        var primary = targets[0];
        var evictable = candidates.Where(model => model.CanEvict && model.DeviceIds.Contains(primary.Id)).ToList();
        var freed = new Dictionary<string, long>(StringComparer.Ordinal);
        foreach (var model in evictable)
        {
            Free(freed, model);
        }

        var fits = Fits(targets, needs, freed, query.ReserveBytes);
        bool othersFit = fits.Skip(1).All(fit => fit.Fits);
        if (!primary.IsGpu || query.SystemMemory is not { } system || !othersFit)
        {
            // CPU-only loads and the further devices of a split cannot offload.
            var all = new Dictionary<string, long>(StringComparer.Ordinal);
            foreach (var model in candidates)
            {
                Free(all, model);
            }

            return new FitReport
            {
                Verdict = FitVerdict.DoesNotFit,
                Footprint = footprint,
                Devices = Fits(targets, needs, all, query.ReserveBytes),
                CpuOffload = query.CpuOffload,
            };
        }

        long layered = footprint.WeightBytes + footprint.StateBytes;
        double primaryShare = footprint.Total <= 0 ? 1 : (double)needs[0] / footprint.Total;
        long primaryLayered = (long)(layered * primaryShare);
        long onGpu = Math.Max(0, fits[0].AvailableBytes - footprint.ActivationBytes);
        long onCpu = Math.Max(0, primaryLayered - onGpu);
        var systemFit = new DeviceFit(system, onCpu, 0, query.ReserveBytes);
        var devices = fits.Select((fit, i) => i == 0 ? fit with { NeededBytes = Math.Min(fit.NeededBytes, fit.AvailableBytes) } : fit).ToList();
        return new FitReport
        {
            Verdict = systemFit.Fits ? FitVerdict.NeedsOffload : FitVerdict.DoesNotFit,
            Footprint = footprint,
            Devices = devices,
            SystemMemory = systemFit,
            Unload = [.. evictable.Select(model => model.Id)],
            CpuShare = layered <= 0 ? 0 : Math.Clamp((double)onCpu / layered, 0, 1),
            CpuOffload = query.CpuOffload,
        };
    }

    /// <summary>Each device's share of <paramref name="total"/>: all of it on one device, else in proportion to the free memory.</summary>
    private static long[] Shares(long total, IReadOnlyList<DeviceMemory> targets, long reserve)
    {
        if (targets.Count == 1)
        {
            return [total];
        }

        var weights = targets.Select(target => SplitWeight(target.FreeBytes, reserve)).ToArray();
        double sum = weights.Sum();
        var shares = weights.Select(weight => (long)(total * (weight / sum))).ToArray();
        shares[^1] += total - shares.Sum();
        return shares;
    }

    private static List<DeviceFit> Fits(IReadOnlyList<DeviceMemory> targets, long[] needs, Dictionary<string, long> freed, long reserve)
        => [.. targets.Select((target, i) => new DeviceFit(target, needs[i], freed.GetValueOrDefault(target.Id), reserve))];

    private static void Free(Dictionary<string, long> freed, ResidentUsage model)
    {
        long each = model.Bytes / Math.Max(1, model.DeviceIds.Count);
        foreach (var id in model.DeviceIds)
        {
            freed[id] = freed.GetValueOrDefault(id) + each;
        }
    }
}
