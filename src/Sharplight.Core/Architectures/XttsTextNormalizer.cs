using System.Text.RegularExpressions;

namespace Sharplight.Architectures;

/// <summary>
/// XTTS v2's English text cleaning (Coqui's <c>multilingual_cleaners</c> for <c>en</c>): quotes removed, lowercase, thousands separators dropped and whole
/// numbers spelled out as num2words does, the abbreviations (mr., dr., st. ...) and symbols (&amp;, @, %, #, $, £, °) written as words, whitespace collapsed. Decimal
/// numbers, ordinals and currency amounts are spelled digit group by digit group rather than as num2words does.
/// </summary>
internal static partial class XttsTextNormalizer
{
    private static readonly (string Abbreviation, string Word)[] s_abbreviations =
    [
        ("mrs", "misess"), ("mr", "mister"), ("dr", "doctor"), ("st", "saint"), ("co", "company"), ("jr", "junior"), ("maj", "major"), ("gen", "general"),
        ("drs", "doctors"), ("rev", "reverend"), ("lt", "lieutenant"), ("hon", "honorable"), ("sgt", "sergeant"), ("capt", "captain"), ("esq", "esquire"),
        ("ltd", "limited"), ("col", "colonel"), ("ft", "fort"),
    ];

    private static readonly (string Symbol, string Word)[] s_symbols =
    [
        ("&", " and "), ("@", " at "), ("%", " percent "), ("#", " hash "), ("$", " dollar "), ("£", " pound "), ("°", " degree "),
    ];

    /// <summary>Cleans English text for the XTTS tokenizer.</summary>
    /// <param name="text">The text.</param>
    public static string English(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        text = text.Replace("\"", "", StringComparison.Ordinal).ToLowerInvariant();
        text = CommaNumber().Replace(text, m => m.Value.Replace(",", "", StringComparison.Ordinal));
        text = Number().Replace(text, m => Num2Words(m.Value));
        foreach (var (abbreviation, word) in s_abbreviations)
        {
            text = Regex.Replace(text, $@"\b{abbreviation}\.", word, RegexOptions.IgnoreCase);
        }

        foreach (var (symbol, word) in s_symbols)
        {
            text = text.Replace(symbol, word, StringComparison.Ordinal).Replace("  ", " ", StringComparison.Ordinal);
        }

        return Whitespace().Replace(text.Trim(), " ");
    }

    [GeneratedRegex(@"\b\d{1,3}(,\d{3})+\b")]
    private static partial Regex CommaNumber();

    [GeneratedRegex(@"[0-9]+")]
    private static partial Regex Number();

    // Whole numbers as num2words spells them: tens and units hyphenated, "and" before a last group below one hundred (one hundred and five).
    private static string Num2Words(string digits)
        => FinalAnd().Replace(TensUnits().Replace(EnglishNumberWords.Spell(digits), "$1-$2"), "$1 and ");

    [GeneratedRegex(@"\b(hundred|thousand|million|billion|trillion) (?=(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(-[a-z]+)?$)")]
    private static partial Regex FinalAnd();

    // num2words joins tens and units with a hyphen (twenty-five).
    [GeneratedRegex(@"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety) (one|two|three|four|five|six|seven|eight|nine)\b")]
    private static partial Regex TensUnits();

    [GeneratedRegex(@"\s+")]
    private static partial Regex Whitespace();
}
