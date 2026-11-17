using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace Sharplight.Speech.Normalization;

/// <summary>
/// Spells out abbreviations, dates, clock times, currency amounts, percentages, ordinals and numbers in English, German, French, Spanish, Italian or
/// Portuguese (the essentials of what the models' own text normalizers, wetext and num2words among them, do for these languages).
/// </summary>
internal sealed class AlphabeticTextNormalizer : ITextNormalizer
{
    private static readonly string[] s_scaleWords =
    [
        "million", "billion", "trillion", "thousand", "Million", "Milliarde", "Mio", "Mrd", "millón", "millones", "milhão", "milhões", "milione", "milioni", "milliard",
        "millions", "bilhões", "miliardo", "miliardi",
    ];

    private readonly AlphabeticLanguage _language;
    private readonly Regex _number;
    private readonly Regex _currencyPrefix;
    private readonly Regex _currencySuffix;
    private readonly Regex _percent;
    private readonly Regex _time;
    private readonly Regex _isoDate = new(@"(?<![\d-])(?<y>\d{4})-(?<m>\d{1,2})-(?<d>\d{1,2})(?![\d-])", RegexOptions.CultureInvariant);
    private readonly Regex _numericDate;
    private readonly Regex _textDate;
    private readonly Regex? _ordinal;
    private readonly (Regex Pattern, string Expansion)[] _abbreviations;
    private readonly string _monthPattern;

    /// <summary>Creates the normalizer of a language.</summary>
    /// <param name="language">The language data.</param>
    internal AlphabeticTextNormalizer(AlphabeticLanguage language)
    {
        _language = language;
        string thousands = "[" + Regex.Escape(language.ThousandsSeparators).Replace("]", "\\]", StringComparison.Ordinal) + "]";
        string decimalSeparator = Regex.Escape(language.DecimalSeparator.ToString());
        string number = $@"\d{{1,3}}(?:{thousands}\d{{3}})+(?:{decimalSeparator}\d+)?|\d+(?:{decimalSeparator}\d+)?";
        var options = RegexOptions.CultureInvariant;
        _number = new Regex($@"(?<![\p{{L}}\d])(?<neg>-(?=\d))?(?<num>{number})(?![\d])", options);
        string symbols = string.Join('|', language.Currencies.Keys.OrderByDescending(k => k.Length).Select(Regex.Escape));
        string scale = string.Join('|', s_scaleWords);
        _currencyPrefix = new Regex($@"(?<![\p{{L}}\d])(?<sym>{symbols})\s?(?<num>{number})(?<scale>\s?(?:{scale})(?![\p{{L}}]))?", options);
        _currencySuffix = new Regex($@"(?<![\p{{L}}\d])(?<num>{number})(?<scale>\s?(?:{scale})(?![\p{{L}}]))?\s?(?<sym>{symbols})(?![\p{{L}}\d])", options);
        _percent = new Regex($@"(?<![\p{{L}}\d])(?<num>{number})\s?%", options);
        _time = new Regex(@"(?<![\d:.])(?<h>[01]?\d|2[0-3]):(?<m>[0-5]\d)(?![\d:])(?:\s?(?<suffix>[ap]m|[ap]\.m\.|Uhr|h(?![\p{L}])))?", RegexOptions.CultureInvariant | RegexOptions.IgnoreCase);
        _numericDate = language.NumericMonthFirst
            ? new Regex(@"(?<![\d/])(?<a>\d{1,2})/(?<b>\d{1,2})/(?<y>\d{4})(?![\d/])", options)
            : new Regex(@"(?<![\d./])(?<a>\d{1,2})[./](?<b>\d{1,2})[./](?<y>\d{4})(?!\d|[./]\d)", options);
        _monthPattern = string.Join('|', language.Months);
        _textDate = TextDate(language, _monthPattern);
        _ordinal = language.OrdinalPattern is { } pattern ? new Regex(pattern, options) : null;
        _abbreviations =
        [
            .. language.Abbreviations.OrderByDescending(a => a.Abbreviation.Length).Select(a =>
                (new Regex(@"(?<![\p{L}\d])" + Regex.Escape(a.Abbreviation).Replace(@"\ ", @"\s?", StringComparison.Ordinal) + @"(?![\p{L}\d])", options), a.Expansion)),
        ];
    }

    /// <inheritdoc/>
    public string Language => _language.Code;

    /// <inheritdoc/>
    public string Normalize(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        foreach (var (pattern, expansion) in _abbreviations)
        {
            text = pattern.Replace(text, expansion);
        }

        text = _textDate.Replace(text, TextDateWords);
        text = _isoDate.Replace(text, match => Date(match, int.Parse(match.Groups["d"].Value, CultureInfo.InvariantCulture), int.Parse(match.Groups["m"].Value, CultureInfo.InvariantCulture)));
        text = _numericDate.Replace(text, NumericDateWords);
        text = _time.Replace(text, TimeWords);
        text = _currencyPrefix.Replace(text, Currency);
        text = _currencySuffix.Replace(text, Currency);
        text = _percent.Replace(text, match => Spell(match.Groups["num"].Value) + " " + _language.Percent);
        if (_ordinal is not null)
        {
            text = _ordinal.Replace(text, match => _language.Words.Ordinal(long.Parse(match.Groups[1].Value, CultureInfo.InvariantCulture)));
        }

        text = _number.Replace(text, match => (match.Groups["neg"].Success ? _language.Minus + " " : "") + Spell(match.Groups["num"].Value));
        foreach (var (symbol, word) in _language.Symbols)
        {
            text = text.Replace(symbol, " " + word + " ", StringComparison.Ordinal);
        }

        return Tidy(text);
    }

    internal static string Tidy(string text)
    {
        text = Regex.Replace(text, @"\s+", " ", RegexOptions.CultureInvariant).Trim();
        return Regex.Replace(text, @"\s+([,.;:!?])", "$1", RegexOptions.CultureInvariant);
    }

    private static Regex TextDate(AlphabeticLanguage language, string months)
    {
        var options = RegexOptions.CultureInvariant | RegexOptions.IgnoreCase;
        return language.Code switch
        {
            "en" => new Regex($@"(?<![\p{{L}}\d])(?:(?<month>{months})\.?\s+(?<day>\d{{1,2}})(?:st|nd|rd|th)?(?![\d:])(?:,?\s+(?<year>\d{{4}})(?!\d))?|(?<day2>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?(?<month2>{months})(?:,?\s+(?<year2>\d{{4}})(?!\d))?)(?![\p{{L}}])", options),
            "de" => new Regex($@"(?<![\p{{L}}\d])(?<day>\d{{1,2}})\.\s?(?<month>{months})(?![\p{{L}}])(?:\s+(?<year>\d{{4}})(?!\d))?", options),
            "es" or "pt" => new Regex($@"(?<![\p{{L}}\d])(?<day>\d{{1,2}})\s+de\s+(?<month>{months})(?![\p{{L}}])(?:\s+de\s+(?<year>\d{{4}})(?!\d))?", options),
            _ => new Regex($@"(?<![\p{{L}}\d])(?<day>\d{{1,2}})(?:er)?\s+(?<month>{months})(?![\p{{L}}])(?:\s+(?<year>\d{{4}})(?!\d))?", options),
        };
    }

    private string TextDateWords(Match match)
    {
        string dayText = match.Groups["day"].Success ? match.Groups["day"].Value : match.Groups["day2"].Value;
        string monthText = match.Groups["month"].Success ? match.Groups["month"].Value : match.Groups["month2"].Value;
        string yearText = match.Groups["year"].Success ? match.Groups["year"].Value : match.Groups["year2"].Value;
        int day = int.Parse(dayText, CultureInfo.InvariantCulture);
        int month = Array.FindIndex(_language.Months, m => string.Equals(m, monthText.TrimEnd('.'), StringComparison.OrdinalIgnoreCase)) + 1;
        if (day is < 1 or > 31 || month == 0)
        {
            return match.Value;
        }

        bool dayFirstEnglish = _language.Code == "en" && match.Groups["day2"].Success;
        return Render(day, month, yearText.Length == 0 ? null : int.Parse(yearText, CultureInfo.InvariantCulture), dayFirstEnglish);
    }

    private string NumericDateWords(Match match)
    {
        int a = int.Parse(match.Groups["a"].Value, CultureInfo.InvariantCulture);
        int b = int.Parse(match.Groups["b"].Value, CultureInfo.InvariantCulture);
        return _language.NumericMonthFirst ? Date(match, b, a) : Date(match, a, b);
    }

    private string Date(Match match, int day, int month)
    {
        if (day is < 1 or > 31 || month is < 1 or > 12)
        {
            return match.Value;
        }

        return Render(day, month, int.Parse(match.Groups["y"].Value, CultureInfo.InvariantCulture), dayFirstEnglish: false);
    }

    private string Render(int day, int month, int? year, bool dayFirstEnglish)
    {
        var language = _language;
        string dayWords = language.DayOrdinal ? language.Words.Ordinal(day) + language.DayOrdinalSuffix
            : day == 1 && language.FirstOfMonth is { } first ? first : language.Words.Cardinal(day);
        string monthWord = language.Months[month - 1];
        if (language.Code is "en" or "de")
        {
            monthWord = char.ToUpperInvariant(monthWord[0]) + monthWord[1..];
        }

        string yearWords = year is { } y ? language.Words.Year(y) : "";
        if (language.Code == "en")
        {
            string head = dayFirstEnglish ? $"{dayWords} of {monthWord}" : $"{monthWord} {dayWords}";
            return yearWords.Length == 0 ? head : $"{head}, {yearWords}";
        }

        var parts = new List<string> { dayWords };
        if (language.DateJoiner.Length > 0)
        {
            parts.Add(language.DateJoiner);
        }

        parts.Add(monthWord);
        if (yearWords.Length > 0)
        {
            if (language.DateJoiner.Length > 0)
            {
                parts.Add(language.DateJoiner);
            }

            parts.Add(yearWords);
        }

        return string.Join(' ', parts);
    }

    private string TimeWords(Match match)
    {
        int hour = int.Parse(match.Groups["h"].Value, CultureInfo.InvariantCulture);
        int minute = int.Parse(match.Groups["m"].Value, CultureInfo.InvariantCulture);
        string suffix = match.Groups["suffix"].Success ? match.Groups["suffix"].Value.Replace(".", "", StringComparison.Ordinal).ToLowerInvariant() : "";
        var words = _language.Words;
        var text = new StringBuilder(words.Cardinal(hour));
        if (_language.Code == "en")
        {
            if (suffix is "am" or "pm")
            {
                text.Append(minute == 0 ? "" : minute < 10 ? " oh " + words.Cardinal(minute) : " " + words.Cardinal(minute)).Append(' ').Append(suffix[0]).Append(" m");
            }
            else if (minute == 0)
            {
                text.Append(hour > 12 || hour == 0 ? " hundred" : " o'clock");
            }
            else
            {
                text.Append(minute < 10 ? " oh " : " ").Append(words.Cardinal(minute));
            }

            return text.ToString();
        }

        if (minute == 0)
        {
            if (_language.HourSuffix.Length > 0)
            {
                text.Append(' ').Append(_language.HourSuffix);
            }

            return text.ToString();
        }

        text.Append(' ').Append(_language.TimeJoiner).Append(' ').Append(words.Cardinal(minute));
        return text.ToString();
    }

    private string Currency(Match match)
    {
        string symbol = match.Groups["sym"].Value;
        if (!_language.Currencies.TryGetValue(symbol, out var names))
        {
            return match.Value;
        }

        string number = match.Groups["num"].Value;
        string scale = match.Groups["scale"].Success ? match.Groups["scale"].Value.Trim() : "";
        var (whole, fraction) = Split(number);
        if (whole is null)
        {
            return Spell(number);
        }

        long units = whole.Value;
        if (scale.Length > 0)
        {
            return $"{Spell(number)} {scale} {names.Many}";
        }

        string text = $"{_language.Words.Cardinal(units)} {(units == 1 ? names.One : names.Many)}";
        if (fraction is { } sub and > 0)
        {
            text += $" {_language.CurrencyJoiner} {_language.Words.Cardinal(sub)} {(sub == 1 ? names.SubOne : names.SubMany)}";
        }

        return text;
    }

    // The whole part and, when the number has one or two decimals, the subunit count; the whole part is null for numbers that do not fit.
    private (long? Whole, long? Fraction) Split(string number)
    {
        string plain = Plain(number, out string fraction);
        if (plain.Length > 15 || fraction.Length > 2)
        {
            return (null, null);
        }

        long whole = long.Parse(plain, CultureInfo.InvariantCulture);
        long? sub = fraction.Length == 0 ? null : long.Parse(fraction.PadRight(2, '0'), CultureInfo.InvariantCulture);
        return (whole, sub);
    }

    // The digits of a written number without its thousands separators, and the digits after the decimal separator.
    private string Plain(string number, out string fraction)
    {
        int point = number.IndexOf(_language.DecimalSeparator, StringComparison.Ordinal);
        fraction = point < 0 ? "" : number[(point + 1)..];
        string whole = point < 0 ? number : number[..point];
        var digits = new StringBuilder(whole.Length);
        foreach (char c in whole)
        {
            if (char.IsAsciiDigit(c))
            {
                digits.Append(c);
            }
        }

        return digits.ToString();
    }

    private string Spell(string number)
    {
        string whole = Plain(number, out string fraction);
        var text = new StringBuilder();
        if (whole.Length > 1 && whole[0] == '0' || whole.Length > 15)
        {
            // Codes, phone numbers and huge values are read digit by digit.
            text.AppendJoin(' ', whole.Select(c => _language.Words.Digit(c - '0')));
        }
        else
        {
            text.Append(_language.Words.Cardinal(long.Parse(whole, CultureInfo.InvariantCulture)));
        }

        if (fraction.Length > 0)
        {
            text.Append(' ').Append(_language.DecimalWord);
            foreach (char c in fraction)
            {
                text.Append(' ').Append(_language.Words.Digit(c - '0'));
            }
        }

        return text.ToString();
    }
}
