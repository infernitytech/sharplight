namespace Sharplight.Cpu;

/// <summary>Best-effort host description read from <c>/proc</c> (Linux); every query degrades to a neutral value elsewhere.</summary>
internal static class HostCpuInfo
{
    /// <summary>The processor model name, or "{architecture} CPU" when unknown.</summary>
    public static string ModelName()
    {
        return ReadCpuInfoField("model name") ?? $"{System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture} CPU";
    }

    /// <summary>The processor vendor id, or an empty string when unknown.</summary>
    public static string Vendor() => ReadCpuInfoField("vendor_id") ?? "";

    /// <summary>Reads a /proc/meminfo field in bytes (Linux); null elsewhere.</summary>
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
