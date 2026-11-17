namespace Sharplight.Speech.Normalization;

/// <summary>German number words (<c>dreiundzwanzig</c>, <c>zweihundertvier</c>, <c>zwei Millionen dreihunderttausend</c>).</summary>
internal sealed class GermanWords : INumberWords
{
    private static readonly string[] s_ones =
    [
        "null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun", "zehn", "elf", "zwölf", "dreizehn", "vierzehn", "fünfzehn", "sechzehn",
        "siebzehn", "achtzehn", "neunzehn",
    ];

    private static readonly string[] s_tens = ["", "", "zwanzig", "dreißig", "vierzig", "fünfzig", "sechzig", "siebzig", "achtzig", "neunzig"];

    // Singular and plural of the scales above a million (feminine nouns: "eine Million", "zwei Millionen").
    private static readonly (string One, string Many)[] s_scales =
        [("", ""), ("", ""), ("Million", "Millionen"), ("Milliarde", "Milliarden"), ("Billion", "Billionen"), ("Billiarde", "Billiarden")];

    /// <inheritdoc/>
    public string Cardinal(long number)
    {
        if (number == 0)
        {
            return "null";
        }

        var words = new List<string>();
        int scale = 0;
        long rest = number;
        while (rest > 0)
        {
            int group = (int)(rest % (scale == 0 ? 1_000_000 : 1000));
            if (group > 0)
            {
                words.Insert(0, scale == 0 ? Below1Million(group) : group == 1 ? "eine " + s_scales[scale + 1].One : $"{Below1000(group, final: false)} {s_scales[scale + 1].Many}");
            }

            rest /= scale == 0 ? 1_000_000 : 1000;
            scale++;
        }

        return string.Join(' ', words);
    }

    /// <inheritdoc/>
    public string Ordinal(long number)
    {
        string cardinal = Cardinal(number);
        if (number < 20)
        {
            return number switch { 1 => "erste", 3 => "dritte", 7 => "siebte", 8 => "achte", _ => cardinal + "te" };
        }

        // The ordinal of the last one or two digits replaces their word (einhunderterste); other numbers take -ste (zwanzigste, einhundertste).
        int last = (int)(number % 100);
        return last is >= 1 and < 20 ? cardinal[..^s_ones[last].Length] + Ordinal(last) : cardinal + "ste";
    }

    /// <inheritdoc/>
    public string Year(long year)
    {
        if (year is >= 1100 and < 2000)
        {
            long high = year / 100;
            long low = year % 100;
            return Cardinal(high) + "hundert" + (low == 0 ? "" : Cardinal(low));
        }

        return Cardinal(year);
    }

    /// <inheritdoc/>
    public string Digit(int digit) => s_ones[digit];

    private static string Below1Million(int n)
    {
        int thousands = n / 1000;
        int rest = n % 1000;
        string text = thousands == 0 ? "" : (thousands == 1 ? "ein" : Below1000(thousands, final: false)) + "tausend";
        return text + (rest == 0 ? "" : Below1000(rest, final: true));
    }

    // A group below one thousand; "ein" instead of "eins" when a noun-like word follows (einhundert, einundzwanzig).
    private static string Below1000(int n, bool final)
    {
        string text = n >= 100 ? (n / 100 == 1 ? "ein" : s_ones[n / 100]) + "hundert" : "";
        int rest = n % 100;
        if (rest == 0)
        {
            return text;
        }

        if (rest < 20)
        {
            return text + (rest == 1 && !final ? "ein" : s_ones[rest]);
        }

        int unit = rest % 10;
        return text + (unit == 0 ? "" : (unit == 1 ? "ein" : s_ones[unit]) + "und") + s_tens[rest / 10];
    }
}
