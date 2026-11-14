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
