namespace Sharplight.Speech.Normalization;

/// <summary>English number words (American style, no <c>and</c>: <c>one hundred twenty-three</c>).</summary>
internal sealed class EnglishWords : INumberWords
{
    private static readonly string[] s_ones =
    [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen",
    ];

    private static readonly string[] s_tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];
    private static readonly string[] s_scales = ["", "thousand", "million", "billion", "trillion", "quadrillion"];

    /// <inheritdoc/>
    public string Cardinal(long number)
    {
        if (number < 20)
        {
            return s_ones[number];
        }

        var parts = new List<string>();
        int scale = 0;
        while (number > 0)
        {
            int group = (int)(number % 1000);
            if (group > 0)
            {
                parts.Insert(0, scale == 0 ? Below1000(group) : $"{Below1000(group)} {s_scales[scale]}");
            }

            number /= 1000;
            scale++;
        }

        return string.Join(' ', parts);
    }

    /// <inheritdoc/>
    public string Ordinal(long number)
    {
        string cardinal = Cardinal(number);
        int cut = Math.Max(cardinal.LastIndexOf(' '), cardinal.LastIndexOf('-')) + 1;
        string last = cardinal[cut..];
        string ordinal = last switch
        {
            "one" => "first",
            "two" => "second",
            "three" => "third",
            "five" => "fifth",
            "eight" => "eighth",
            "nine" => "ninth",
            "twelve" => "twelfth",
            _ when last.EndsWith('y') => last[..^1] + "ieth",
            _ => last + "th",
        };
        return cardinal[..cut] + ordinal;
    }

    /// <inheritdoc/>
    public string Year(long year)
    {
        if (year is >= 1100 and < 2000 or >= 2010 and < 2100)
        {
            long high = year / 100;
            long low = year % 100;
            return low == 0 ? $"{Cardinal(high)} hundred" : low < 10 ? $"{Cardinal(high)} oh {Cardinal(low)}" : $"{Cardinal(high)} {Cardinal(low)}";
        }

        return Cardinal(year);
    }

    /// <inheritdoc/>
    public string Digit(int digit) => s_ones[digit];

    private static string Below1000(int n)
    {
        string text = n >= 100 ? s_ones[n / 100] + " hundred" : "";
        int rest = n % 100;
        if (rest == 0)
        {
            return text;
        }

        string tail = rest < 20 ? s_ones[rest] : s_tens[rest / 10] + (rest % 10 == 0 ? "" : "-" + s_ones[rest % 10]);
        return text.Length == 0 ? tail : text + " " + tail;
    }
}
