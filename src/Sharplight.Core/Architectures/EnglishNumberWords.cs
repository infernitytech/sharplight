using System.Globalization;
using System.Text;

namespace Sharplight.Architectures;

/// <summary>Spells numbers as English words (the text normalization of OuteTTS: <c>42</c> becomes <c>forty two</c>, <c>3.14</c> <c>three point one four</c>).</summary>
internal static class EnglishNumberWords
{
    private static readonly string[] s_ones =
    [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
    ];

    private static readonly string[] s_tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];

    private static readonly string[] s_scales = ["", "thousand", "million", "billion", "trillion"];

    /// <summary>Spells a non-negative decimal number given as digits with an optional fractional part (<c>12</c>, <c>0.5</c>).</summary>
    /// <param name="number">The digits.</param>
    public static string Spell(string number)
    {
        int point = number.IndexOf('.', StringComparison.Ordinal);
        string whole = point < 0 ? number : number[..point];
        var result = new StringBuilder(Integer(whole));
        if (point >= 0 && point + 1 < number.Length)
        {
            result.Append(" point");
            foreach (char digit in number[(point + 1)..])
            {
                result.Append(' ').Append(s_ones[digit - '0']);
            }
        }

        return result.ToString();
    }

    private static string Integer(string digits)
    {
        digits = digits.TrimStart('0');
        if (digits.Length == 0)
        {
            return s_ones[0];
        }

        if (digits.Length > 15)
        {
            return string.Join(' ', digits.Select(d => s_ones[d - '0']));
        }

        long value = long.Parse(digits, CultureInfo.InvariantCulture);
        var parts = new List<string>();
        for (int scale = 0; value > 0; scale++, value /= 1000)
        {
            int group = (int)(value % 1000);
            if (group > 0)
            {
                parts.Insert(0, (Group(group) + " " + s_scales[scale]).Trim());
            }
        }

        return string.Join(' ', parts);
    }

    private static string Group(int value)
    {
        var words = new List<string>();
        if (value >= 100)
        {
            words.Add(s_ones[value / 100] + " hundred");
            value %= 100;
        }

        if (value >= 20)
        {
            words.Add(s_tens[value / 10]);
            value %= 10;
            if (value > 0)
            {
                words.Add(s_ones[value]);
            }
        }
        else if (value > 0)
        {
            words.Add(s_ones[value]);
        }

        return string.Join(' ', words);
    }
}
