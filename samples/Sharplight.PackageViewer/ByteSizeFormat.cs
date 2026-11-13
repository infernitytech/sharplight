using System.Globalization;

namespace Sharplight.PackageViewer;

/// <summary>Formats byte counts for display with binary units (<c>1.5 MiB</c>).</summary>
internal static class ByteSizeFormat
{
    private static readonly string[] s_units = ["B", "KiB", "MiB", "GiB", "TiB"];
    private const double UnitStep = 1024;

    public static string Format(long bytes)
    {
        double value = bytes;
        int unit = 0;
        while (Math.Abs(value) >= UnitStep && unit < s_units.Length - 1)
        {
            value /= UnitStep;
            unit++;
        }

        var format = unit == 0 ? "0" : "0.0#";
        return $"{value.ToString(format, CultureInfo.InvariantCulture)} {s_units[unit]}";
    }

    /// <summary>The exact count with thousands separators and the rounded size, e.g. <c>1,048,576 (1.0 MiB)</c>.</summary>
    public static string FormatExact(long bytes) =>
        bytes < UnitStep ? Format(bytes) : $"{bytes.ToString("N0", CultureInfo.InvariantCulture)} B ({Format(bytes)})";
}
