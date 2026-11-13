using System.Globalization;

namespace Sharplight.Hub;

/// <summary>Formats byte counts for log messages with binary units (<c>1.50 GiB</c>).</summary>
internal static class ByteCount
{
    private static readonly string[] s_units = ["B", "KiB", "MiB", "GiB", "TiB"];

    public static string Format(long bytes)
    {
        double value = bytes;
        int unit = 0;
        while (value >= 1024 && unit < s_units.Length - 1)
        {
            value /= 1024;
            unit++;
        }

        return unit == 0
            ? string.Create(CultureInfo.InvariantCulture, $"{bytes} B")
            : string.Create(CultureInfo.InvariantCulture, $"{value:F2} {s_units[unit]}");
    }
}
