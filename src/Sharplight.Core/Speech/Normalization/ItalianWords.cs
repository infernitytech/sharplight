namespace Sharplight.Speech.Normalization;

/// <summary>Italian number words (<c>ventitré</c>, <c>centottanta</c>, <c>duemila</c>, <c>due milioni</c>).</summary>
internal sealed class ItalianWords : INumberWords
{
    private static readonly string[] s_ones =
    [
        "zero", "uno", "due", "tre", "quattro", "cinque", "sei", "sette", "otto", "nove", "dieci", "undici", "dodici", "tredici", "quattordici", "quindici",
        "sedici", "diciassette", "diciotto", "diciannove",
    ];

    private static readonly string[] s_tens = ["", "", "venti", "trenta", "quaranta", "cinquanta", "sessanta", "settanta", "ottanta", "novanta"];
    private static readonly string[] s_ordinals = ["", "primo", "secondo", "terzo", "quarto", "quinto", "sesto", "settimo", "ottavo", "nono", "decimo"];

    /// <inheritdoc/>
    public string Cardinal(long number)
    {
        if (number == 0)
        {
            return "zero";
        }

        // The accent of a final tré (ventitré) belongs to the last word when it is a compound.
        string text = Compose(number);
        int space = text.LastIndexOf(' ') + 1;
        return text.Length - space > 3 && text.EndsWith("tre", StringComparison.Ordinal) ? text[..^1] + "é" : text;
    }

    /// <inheritdoc/>
    public string Ordinal(long number)
    {
        if (number <= 10)
        {
            return s_ordinals[number];
        }

        string cardinal = Cardinal(number);
        if (cardinal.EndsWith("mila", StringComparison.Ordinal))
        {
            return cardinal[..^4] + "millesimo";
        }

        if (cardinal.EndsWith("tré", StringComparison.Ordinal))
        {
            return cardinal[..^1] + "eesimo";
        }

        if (cardinal.EndsWith("tre", StringComparison.Ordinal) || cardinal.EndsWith("sei", StringComparison.Ordinal))
        {
            return cardinal + "esimo";
        }

        return (cardinal.EndsWith('e') || cardinal.EndsWith('o') || cardinal.EndsWith('a') || cardinal.EndsWith('i') ? cardinal[..^1] : cardinal) + "esimo";
    }

    /// <inheritdoc/>
    public string Year(long year) => Cardinal(year);

    /// <inheritdoc/>
    public string Digit(int digit) => s_ones[digit];

    // Everything below a million is one word (centoventitré, duemilatrecento); millions and above are separate words.
    private static string Compose(long n)
    {
        if (n < 1000)
        {
            return Below1000((int)n);
        }

        if (n < 1_000_000)
        {
            int thousands = (int)(n / 1000);
            int rest = (int)(n % 1000);
            string head = thousands == 1 ? "mille" : Below1000(thousands) + "mila";
            return head + (rest == 0 ? "" : Below1000(rest));
        }

        long scale = n < 1_000_000_000 ? 1_000_000 : n < 1_000_000_000_000 ? 1_000_000_000 : 1_000_000_000_000;
        long high = n / scale;
        long remainder = n % scale;
        (string one, string many) = scale switch
        {
            1_000_000 => ("un milione", "milioni"),
            1_000_000_000 => ("un miliardo", "miliardi"),
            _ => ("un bilione", "bilioni"),
        };
        string text = high == 1 ? one : Compose(high) + " " + many;
        return remainder == 0 ? text : text + " " + Compose(remainder);
    }

    private static string Below1000(int n)
    {
        string text = n >= 100 ? (n / 100 == 1 ? "" : s_ones[n / 100]) + "cento" : "";
        int rest = n % 100;
        if (rest == 0)
        {
            return text;
        }

        string tail;
        if (rest < 20)
        {
            tail = s_ones[rest];
        }
        else
        {
            int ten = rest / 10;
            int unit = rest % 10;
            string tens = s_tens[ten];
            // The tens lose their last vowel before uno and otto (ventuno, ventotto).
            tail = unit == 0 ? tens : (unit is 1 or 8 ? tens[..^1] : tens) + s_ones[unit];
        }

        if (text.EndsWith("cento", StringComparison.Ordinal) && tail.StartsWith("ottanta", StringComparison.Ordinal))
        {
            text = text[..^1];
        }

        return text + tail;
    }
}
