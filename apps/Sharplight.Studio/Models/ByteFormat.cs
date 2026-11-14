using System.Globalization;

namespace Sharplight.Studio.Models;

/// <summary>Formats byte counts, rates and durations for display.</summary>
internal static class ByteFormat
{
    private static readonly string[] s_units = ["B", "KB", "MB", "GB", "TB"];

    /// <summary><c>1.5 GB</c>-style size (binary multiples).</summary>
    public static string Format(long bytes)
    {
        double value = Math.Max(0, bytes);
        int unit = 0;
        while (value >= 1024 && unit < s_units.Length - 1)
        {
            value /= 1024;
            unit++;
        }

        string format = unit == 0 || value >= 100 ? "F0" : "F1";
        return value.ToString(format, CultureInfo.InvariantCulture) + " " + s_units[unit];
    }

    /// <summary><c>12.3 MB/s</c>.</summary>
    public static string Rate(double bytesPerSecond) => Format((long)bytesPerSecond) + "/s";

    /// <summary><c>1h 02m</c>, <c>3m 05s</c> or <c>12s</c>.</summary>
    public static string Duration(TimeSpan duration)
    {
        if (duration.TotalHours >= 1)
        {
            return $"{(int)duration.TotalHours}h {duration.Minutes:D2}m";
        }

        return duration.TotalMinutes >= 1 ? $"{(int)duration.TotalMinutes}m {duration.Seconds:D2}s" : $"{Math.Max(0, (int)duration.TotalSeconds)}s";
    }

    /// <summary><c>1.2M</c>, <c>34.5k</c> or <c>812</c>.</summary>
    public static string Count(double value)
    {
        return value switch
        {
            >= 1e9 => (value / 1e9).ToString("F1", CultureInfo.InvariantCulture) + "B",
            >= 1e6 => (value / 1e6).ToString("F1", CultureInfo.InvariantCulture) + "M",
            >= 1e4 => (value / 1e3).ToString("F1", CultureInfo.InvariantCulture) + "k",
            _ => value.ToString("F0", CultureInfo.InvariantCulture),
        };
    }
}
