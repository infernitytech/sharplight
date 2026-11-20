using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace Sharplight.Cpu;

/// <summary>
/// Best-effort host description read from <c>/proc</c> (Linux) or <c>sysctl</c> (macOS); every query degrades to a neutral
/// value elsewhere.
/// </summary>
internal static unsafe partial class HostCpuInfo
{
    /// <summary>The processor model name, or "{architecture} CPU" when unknown.</summary>
    public static string ModelName()
    {
        string? name = OperatingSystem.IsMacOS() ? SysctlString("machdep.cpu.brand_string") : ReadCpuInfoField("model name");
        return name ?? $"{RuntimeInformation.ProcessArchitecture} CPU";
    }

    /// <summary>The processor vendor id, or an empty string when unknown.</summary>
    public static string Vendor() => ReadCpuInfoField("vendor_id") ?? "";

    /// <summary>
    /// Host memory this process may use, in bytes: the smaller of the machine's memory and the memory limit of its control group
    /// (containers: <c>free</c> and /proc/meminfo show the whole machine, the OOM killer enforces the cgroup limit). Null when unknown.
    /// </summary>
    public static long? TotalMemory()
    {
        long? machine = MemInfo("MemTotal");
        long? limit = CgroupLimit()?.Limit;
        return machine is null ? limit : limit is null ? machine : Math.Min(machine.Value, limit.Value);
    }

    /// <summary>
    /// Host memory this process can still allocate, in bytes: /proc/meminfo's MemAvailable, bounded by what the control group's limit
    /// leaves (its usage minus reclaimable file cache). Null when unknown.
    /// </summary>
    public static long? AvailableMemory()
    {
        long? machine = MemInfo("MemAvailable");
        if (CgroupLimit() is not { } cgroup)
        {
            return machine;
        }

        long left = Math.Max(0, cgroup.Limit - cgroup.Usage + cgroup.Reclaimable);
        return machine is null ? left : Math.Min(machine.Value, left);
    }

    /// <summary>The tightest memory limit of this process's cgroup (v2, or v1) with its usage and reclaimable file cache; null without one.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static (long Limit, long Usage, long Reclaimable)? CgroupLimit()
    {
        try
        {
            if (!OperatingSystem.IsLinux())
            {
                return null;
            }

            // cgroup v2: the process's own group and its ancestors; the tightest limit applies.
            string? relative = File.Exists("/proc/self/cgroup")
                ? File.ReadLines("/proc/self/cgroup").FirstOrDefault(line => line.StartsWith("0::", StringComparison.Ordinal))?[3..]
                : null;
            if (relative is not null)
            {
                (long, long, long)? best = null;
                for (string? dir = "/sys/fs/cgroup" + relative.TrimEnd('/'); dir is not null && dir.StartsWith("/sys/fs/cgroup", StringComparison.Ordinal); dir = Path.GetDirectoryName(dir))
                {
                    string max = Path.Combine(dir, "memory.max");
                    if (File.Exists(max) && long.TryParse(File.ReadAllText(max).Trim(), out long limit))
                    {
                        long usage = ReadLong(Path.Combine(dir, "memory.current")) ?? 0;
                        long reclaimable = ReadStat(Path.Combine(dir, "memory.stat"), "inactive_file") ?? 0;
                        if (best is null || limit - usage + reclaimable < best.Value.Item1 - best.Value.Item2 + best.Value.Item3)
                        {
                            best = (limit, usage, reclaimable);
                        }
                    }
                }

                if (best is not null)
                {
                    return best;
                }
            }

            // cgroup v1: an unlimited group reports a huge limit.
            long? v1 = ReadLong("/sys/fs/cgroup/memory/memory.limit_in_bytes");
            if (v1 is { } v1Limit && v1Limit < long.MaxValue / 2)
            {
                return (v1Limit, ReadLong("/sys/fs/cgroup/memory/memory.usage_in_bytes") ?? 0,
                    ReadStat("/sys/fs/cgroup/memory/memory.stat", "total_inactive_file") ?? 0);
            }
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or FormatException)
        {
            // Informational only: an unreadable cgroup file leaves the machine's figures.
        }

        return null;
    }

    private static long? ReadLong(string path) => File.Exists(path) && long.TryParse(File.ReadAllText(path).Trim(), out long value) ? value : null;

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static long? ReadStat(string path, string key)
    {
        if (!File.Exists(path))
        {
            return null;
        }

        foreach (var line in File.ReadLines(path))
        {
            if (line.StartsWith(key + " ", StringComparison.Ordinal) && long.TryParse(line.AsSpan(key.Length + 1), out long value))
            {
                return value;
            }
        }

        return null;
    }

    /// <summary>Reads a /proc/meminfo field in bytes (Linux); null elsewhere.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static long? MemInfo(string key)
    {
        try
        {
            if (!File.Exists("/proc/meminfo"))
            {
                return null;
            }

            foreach (var line in File.ReadLines("/proc/meminfo"))
            {
                if (line.StartsWith(key + ":", StringComparison.Ordinal))
                {
                    var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                    return long.Parse(parts[1]) * 1024;
                }
            }
        }
        catch
        {
            // Informational only: an unreadable /proc must not prevent the provider from working.
        }

        return null;
    }

    /// <summary>A string value of <c>sysctlbyname</c> (macOS), or null when it cannot be read.</summary>
    private static string? SysctlString(string name)
    {
        try
        {
            nuint size = 0;
            if (SysctlByName(name, null, &size, null, 0) != 0 || size == 0 || size > 4096)
            {
                return null;
            }

            byte* buffer = stackalloc byte[(int)size];
            if (SysctlByName(name, buffer, &size, null, 0) != 0)
            {
                return null;
            }

            string value = Encoding.UTF8.GetString(buffer, (int)size).TrimEnd('\0').Trim();
            return value.Length > 0 ? value : null;
        }
        catch (Exception e) when (e is DllNotFoundException or EntryPointNotFoundException)
        {
            return null;
        }
    }

    [LibraryImport("libc", EntryPoint = "sysctlbyname", StringMarshalling = StringMarshalling.Utf8)]
    private static partial int SysctlByName(string name, void* value, nuint* size, void* newValue, nuint newSize);

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static string? ReadCpuInfoField(string field)
    {
        try
        {
            if (!File.Exists("/proc/cpuinfo"))
            {
                return null;
            }

            foreach (var line in File.ReadLines("/proc/cpuinfo"))
            {
                if (!line.StartsWith(field, StringComparison.Ordinal))
                {
                    continue;
                }

                int colon = line.IndexOf(':');
                if (colon >= 0)
                {
                    return line[(colon + 1)..].Trim();
                }
            }
        }
        catch
        {
            // Informational only: an unreadable /proc must not prevent the provider from working.
        }

        return null;
    }
}
