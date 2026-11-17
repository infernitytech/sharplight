namespace Sharplight.Speech.Normalization;

/// <summary>French number words (<c>soixante et onze</c>, <c>quatre-vingt-dix-neuf</c>, <c>deux cents</c>).</summary>
internal sealed class FrenchWords : INumberWords
{
    private static readonly string[] s_ones =
    [
        "zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix", "onze", "douze", "treize", "quatorze", "quinze", "seize",
        "dix-sept", "dix-huit", "dix-neuf",
    ];

    private static readonly string[] s_tens = ["", "", "vingt", "trente", "quarante", "cinquante", "soixante"];

    /// <inheritdoc/>
    public string Cardinal(long number)
    {
        if (number == 0)
        {
            return "zéro";
        }

        var words = new List<string>();
        long rest = number;
        string[] scaleOne = ["", "mille", "million", "milliard", "billion"];
        string[] scaleMany = ["", "mille", "millions", "milliards", "billions"];
        int scale = 0;
        while (rest > 0)
        {
            int group = (int)(rest % 1000);
            if (group > 0)
            {
                if (scale == 0)
                {
                    words.Insert(0, Below1000(group, plural: true));
                }
                else if (scale == 1)
                {
                    words.Insert(0, group == 1 ? "mille" : Below1000(group, plural: false) + " mille");
                }
                else
                {
                    words.Insert(0, group == 1 ? "un " + scaleOne[scale] : Below1000(group, plural: false) + " " + scaleMany[scale]);
                }
            }

            rest /= 1000;
            scale++;
        }

        return string.Join(' ', words);
    }

    /// <inheritdoc/>
    public string Ordinal(long number)
    {
        if (number == 1)
        {
            return "premier";
        }

        string cardinal = Cardinal(number);
        if (cardinal.EndsWith("quatre-vingts", StringComparison.Ordinal) || cardinal.EndsWith("cents", StringComparison.Ordinal))
        {
            cardinal = cardinal[..^1];
        }

        string stem = cardinal switch
        {
            _ when cardinal.EndsWith('e') => cardinal[..^1],
            _ when cardinal.EndsWith("cinq", StringComparison.Ordinal) => cardinal + "u",
            _ when cardinal.EndsWith("neuf", StringComparison.Ordinal) => cardinal[..^1] + "v",
            _ => cardinal,
        };
        return stem + "ième";
    }

    /// <inheritdoc/>
    public string Year(long year) => Cardinal(year);

    /// <inheritdoc/>
    public string Digit(int digit) => s_ones[digit];

    // A group below one thousand; "plural" gives the s of "cents" and "quatre-vingts" when nothing follows (the last group, or before a scale word that is not mille).
    private static string Below1000(int n, bool plural)
    {
        int hundreds = n / 100;
        int rest = n % 100;
        string text = hundreds switch
        {
            0 => "",
            1 => "cent",
            _ => s_ones[hundreds] + " cent" + (rest == 0 && plural ? "s" : ""),
        };
        if (rest == 0)
        {
            return text;
        }

        string tail = Below100(rest, plural);
        return text.Length == 0 ? tail : text + " " + tail;
    }

    private static string Below100(int n, bool plural)
    {
        if (n < 20)
        {
            return s_ones[n];
        }

        int ten = n / 10;
        int unit = n % 10;
        switch (ten)
        {
            case 7:
                return unit == 1 ? "soixante et onze" : "soixante-" + s_ones[10 + unit];
            case 8:
                return unit == 0 ? "quatre-vingt" + (plural ? "s" : "") : "quatre-vingt-" + s_ones[unit];
            case 9:
                return "quatre-vingt-" + s_ones[10 + unit];
            default:
                return unit == 0 ? s_tens[ten] : unit == 1 ? s_tens[ten] + " et un" : s_tens[ten] + "-" + s_ones[unit];
        }
    }
}
