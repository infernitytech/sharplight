using System.Globalization;

namespace Sharplight.Hosting;

/// <summary>Byte sizes as written in options and on command lines: a number of bytes with an optional K, M, G or T suffix (powers of 1024).</summary>
public static class MemorySize
{
    /// <summary>Parses <paramref name="text"/> (<c>"1073741824"</c>, <c>"512M"</c>, <c>"1.5G"</c>, <c>"2GiB"</c>, <c>"2GB"</c>).</summary>
    /// <param name="text">The size.</param>
    /// <returns>The size in bytes.</returns>
    /// <exception cref="FormatException">The text is not a size.</exception>
    public static long Parse(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        var value = text.Trim();
        foreach (var unit in new[] { "IB", "B" })
        {
            if (value.Length > unit.Length && value.EndsWith(unit, StringComparison.OrdinalIgnoreCase) && char.IsLetter(value[^(unit.Length + 1)]))
            {
                value = value[..^unit.Length];
                break;
            }
        }

        long multiplier = 1;
        if (value.Length > 0 && char.IsLetter(value[^1]))
        {
            multiplier = char.ToUpperInvariant(value[^1]) switch
            {
                'K' => 1L << 10,
                'M' => 1L << 20,
                'G' => 1L << 30,
                'T' => 1L << 40,
                _ => throw new FormatException($"'{text}' is not a size (expected a number with an optional K, M, G or T suffix)."),
            };
            value = value[..^1];
        }

        if (!double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out var number) || number < 0)
        {
            throw new FormatException($"'{text}' is not a size (expected a number with an optional K, M, G or T suffix).");
        }

        return (long)(number * multiplier);
    }

    /// <summary>Formats <paramref name="bytes"/> for people (<c>1.5 GiB</c>).</summary>
    /// <param name="bytes">The size in bytes.</param>
    /// <returns>The text.</returns>
    public static string Format(long bytes)
    {
        string[] units = ["B", "KiB", "MiB", "GiB", "TiB"];
        double value = bytes;
        int unit = 0;
        while (Math.Abs(value) >= 1024 && unit < units.Length - 1)
        {
            value /= 1024;
            unit++;
        }

        return unit == 0 ? $"{bytes} B" : string.Create(CultureInfo.InvariantCulture, $"{value:0.#} {units[unit]}");
    }
}
