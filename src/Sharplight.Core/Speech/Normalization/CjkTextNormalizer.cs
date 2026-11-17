using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace Sharplight.Speech.Normalization;

/// <summary>Spells out numbers, dates, clock times, currency amounts and percentages in Chinese or Japanese (the essentials of wetext's Chinese and Japanese rules).</summary>
internal sealed partial class CjkTextNormalizer(CjkLanguage language) : ITextNormalizer
{
    /// <inheritdoc/>
    public string Language => language.Code;

    /// <inheritdoc/>
    public string Normalize(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        text = FullWidth(text);
        text = FullDate().Replace(text, match => Year(match.Groups["y"].Value) + "年" + language.Words.Cardinal(Number(match.Groups["m"])) + "月" + language.Words.Cardinal(Number(match.Groups["d"]))
            + (match.Groups["dc"].Length > 0 ? match.Groups["dc"].Value : "日"));
        text = YearOnly().Replace(text, match => Year(match.Groups["y"].Value) + "年");
        text = Time().Replace(text, TimeWords);
        text = CurrencyPrefix().Replace(text, Currency);
        text = Percent().Replace(text, match => language.PercentPrefix + Spell(match.Groups["num"].Value) + language.PercentSuffix);
        text = Numbers().Replace(text, match => (match.Groups["neg"].Success ? language.Minus : "") + Spell(match.Groups["num"].Value, match.Index + match.Length, text));
        foreach (var (symbol, word) in language.Symbols)
        {
            text = text.Replace(symbol, word, StringComparison.Ordinal);
        }

        return AlphabeticTextNormalizer.Tidy(text);
    }

    [GeneratedRegex(@"(?<![\d])(?<y>\d{4})(?:年|[-/.])(?<m>\d{1,2})(?:月|[-/.])(?<d>\d{1,2})(?<dc>[日号]?)", RegexOptions.CultureInvariant)]
    private static partial Regex FullDate();

    [GeneratedRegex(@"(?<![\d.])(?<y>\d{4})年", RegexOptions.CultureInvariant)]
    private static partial Regex YearOnly();

    [GeneratedRegex(@"(?<![\d:.])(?<h>[01]?\d|2[0-3]):(?<m>[0-5]\d)(?![\d:])", RegexOptions.CultureInvariant)]
    private static partial Regex Time();

    [GeneratedRegex(@"(?<sym>[¥￥$€£])\s?(?<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)", RegexOptions.CultureInvariant)]
    private static partial Regex CurrencyPrefix();

    [GeneratedRegex(@"(?<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s?%", RegexOptions.CultureInvariant)]
    private static partial Regex Percent();

    [GeneratedRegex(@"(?<![\d.])(?<neg>-(?=\d))?(?<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)", RegexOptions.CultureInvariant)]
    private static partial Regex Numbers();

    // The measure words after which Mandarin reads 2 as 两.
    private static bool IsMeasure(char c) => "个位只本天人次张条件块双份把台辆间".Contains(c, StringComparison.Ordinal);

    private static long Number(Group group) => long.Parse(group.Value, CultureInfo.InvariantCulture);

    private static string FullWidth(string text)
    {
        var builder = new StringBuilder(text.Length);
        foreach (char c in text)
        {
            builder.Append(c is >= '０' and <= '９' ? (char)(c - '０' + '0') : c is '％' ? '%' : c is '：' ? ':' : c);
        }

        return builder.ToString();
    }

    private string Year(string digits) => language.YearByDigits ? language.Words.Year(long.Parse(digits, CultureInfo.InvariantCulture)) : language.Words.Cardinal(long.Parse(digits, CultureInfo.InvariantCulture));

    private string TimeWords(Match match)
    {
        int hour = int.Parse(match.Groups["h"].Value, CultureInfo.InvariantCulture);
        int minute = int.Parse(match.Groups["m"].Value, CultureInfo.InvariantCulture);
        string text = language.Words.Cardinal(hour) + language.HourUnit;
        if (minute == 0)
        {
            return text;
        }

        return text + (minute < 10 && language.Code == "zh" ? "零" : "") + language.Words.Cardinal(minute) + language.MinuteUnit;
    }

    private string Currency(Match match)
    {
        char symbol = match.Groups["sym"].Value[0];
        string unit = language.Currencies[symbol];
        string number = match.Groups["num"].Value.Replace(",", "", StringComparison.Ordinal);
        int point = number.IndexOf('.', StringComparison.Ordinal);
        if (point < 0)
        {
            return Spell(number) + unit;
        }

        string fraction = number[(point + 1)..];
        if (language.Code == "zh" && fraction.Length <= 2)
        {
            string text = Spell(number[..point]) + unit;
            fraction = fraction.PadRight(2, '0');
            if (fraction[0] != '0')
            {
                text += language.Words.Digit(fraction[0] - '0') + "角";
            }

            if (fraction[1] != '0')
            {
                text += (fraction[0] == '0' ? "零" : "") + language.Words.Digit(fraction[1] - '0') + "分";
            }

            return text;
        }

        if (language.Code == "ja" && symbol is '$' && fraction.Length <= 2)
        {
            long cents = long.Parse(fraction.PadRight(2, '0'), CultureInfo.InvariantCulture);
            return Spell(number[..point]) + unit + (cents == 0 ? "" : language.Words.Cardinal(cents) + "セント");
        }

        return Spell(number) + unit;
    }

    private string Spell(string number, int end = -1, string? text = null)
    {
        number = number.Replace(",", "", StringComparison.Ordinal);
        int point = number.IndexOf('.', StringComparison.Ordinal);
        string whole = point < 0 ? number : number[..point];
        var result = new StringBuilder();
        if (whole.Length > 1 && whole[0] == '0' || whole.Length > 15)
        {
            result.AppendJoin("", whole.Select(c => language.Words.Digit(c - '0')));
        }
        else
        {
            long value = long.Parse(whole, CultureInfo.InvariantCulture);
            string words = language.Words.Cardinal(value);
            if (language.TwoBeforeMeasures && value == 2 && point < 0 && end >= 0 && text is not null && end < text.Length && IsMeasure(text[end]))
            {
                words = "两";
            }

            result.Append(words);
        }

        if (point >= 0)
        {
            result.Append(language.DecimalWord);
            foreach (char c in number[(point + 1)..])
            {
                result.Append(language.Words.Digit(c - '0'));
            }
        }

        return result.ToString();
    }
}
