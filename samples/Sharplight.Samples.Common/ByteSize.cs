using System.Globalization;

namespace Sharplight.Samples;

/// <summary>Parses and formats memory sizes such as <c>512M</c>, <c>1.5G</c> or <c>2GiB</c> (binary multiples).</summary>
public static class ByteSize
{
    private const int KibiShift = 10;
    private const int MebiShift = 20;
    private const int GibiShift = 30;
    private const int TebiShift = 40;

    /// <summary>Parses <paramref name="text"/>; a bare number is a byte count and the suffixes K, M, G, T (optionally followed by <c>i</c> and/or <c>B</c>) are powers of 1024.</summary>
    /// <exception cref="FormatException">The text is not a size.</exception>
    public static long Parse(string text)
    {
        var number = text.Trim().TrimEnd('B', 'b').TrimEnd('i', 'I');
        int shift = 0;
        if (number.Length > 0 && TryGetShift(number[^1], out int unitShift))
        {
            shift = unitShift;
            number = number[..^1];
        }

        if (!double.TryParse(number, NumberStyles.Float, CultureInfo.InvariantCulture, out double value) || value < 0)
        {
            throw new FormatException($"'{text}' is not a size; expected a number with an optional K, M, G or T suffix (for example 512M or 1.5G).");
        }

        return (long)(value * (1L << shift));
    }

    /// <summary>Formats a byte count with a binary unit, for example <c>1.50 GiB</c>.</summary>
    public static string Format(long bytes)
    {
        return bytes switch
        {
            >= 1L << TebiShift => $"{bytes / (double)(1L << TebiShift):F2} TiB",
            >= 1L << GibiShift => $"{bytes / (double)(1L << GibiShift):F2} GiB",
            >= 1L << MebiShift => $"{bytes / (double)(1L << MebiShift):F1} MiB",
            >= 1L << KibiShift => $"{bytes / (double)(1L << KibiShift):F1} KiB",
            _ => $"{bytes} B",
        };
    }

    private static bool TryGetShift(char suffix, out int shift)
    {
        shift = char.ToUpperInvariant(suffix) switch
        {
            'K' => KibiShift,
            'M' => MebiShift,
            'G' => GibiShift,
            'T' => TebiShift,
            _ => 0,
        };
        return shift != 0;
    }
}
