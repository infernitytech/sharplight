namespace Sharplight.Speech.Normalization;

/// <summary>The words and conventions one alphabetic language's <see cref="AlphabeticTextNormalizer"/> needs.</summary>
internal sealed class AlphabeticLanguage
{
    /// <summary>ISO 639-1 code.</summary>
    public required string Code { get; init; }

    /// <summary>The number words.</summary>
    public required INumberWords Words { get; init; }

    /// <summary>The decimal separator of written numbers.</summary>
    public required char DecimalSeparator { get; init; }

    /// <summary>The characters that separate thousands groups (a group is exactly three digits).</summary>
    public required string ThousandsSeparators { get; init; }

    /// <summary>The word read for the decimal separator (<c>point</c>, <c>Komma</c>).</summary>
    public required string DecimalWord { get; init; }

    /// <summary>The word for a minus sign.</summary>
    public required string Minus { get; init; }

    /// <summary>The words for a percent sign.</summary>
    public required string Percent { get; init; }

    /// <summary>The twelve month names, lowercase as written in the language's dates.</summary>
    public required string[] Months { get; init; }

    /// <summary>The currencies by symbol and code: names of the unit and the subunit, singular and plural.</summary>
    public required IReadOnlyDictionary<string, CurrencyNames> Currencies { get; init; }

    /// <summary>Abbreviations with their expansions (matched exactly, case sensitive).</summary>
    public required (string Abbreviation, string Expansion)[] Abbreviations { get; init; }

    /// <summary>Symbols with the words read for them.</summary>
    public required (string Symbol, string Word)[] Symbols { get; init; }

    /// <summary>The word that joins the parts of a date (<c>de</c> in Spanish and Portuguese), or empty.</summary>
    public string DateJoiner { get; init; } = "";

    /// <summary>The month comes before the day in dates (English).</summary>
    public bool MonthFirst { get; init; }

    /// <summary>The numeric date order of <c>a/b/yyyy</c>: the month first (US English) or the day first.</summary>
    public bool NumericMonthFirst { get; init; }

    /// <summary>The day of a date is read as an ordinal (English, German); otherwise as a cardinal with an ordinal for the first.</summary>
    public bool DayOrdinal { get; init; }

    /// <summary>The ending added to an ordinal day in German dates (<c>dritten</c>), or empty.</summary>
    public string DayOrdinalSuffix { get; init; } = "";

    /// <summary>The ordinal indicator pattern after digits (a regular expression with the digits in group 1), or <see langword="null"/>.</summary>
    public string? OrdinalPattern { get; init; }

    /// <summary>The word between the hour and the minutes of a clock time (<c>Uhr</c>, <c>heures</c>, <c>y</c>).</summary>
    public required string TimeJoiner { get; init; }

    /// <summary>The word after an exact hour (<c>o'clock</c>, <c>Uhr</c>, <c>en punto</c>), or empty.</summary>
    public string HourSuffix { get; init; } = "";

    /// <summary>The cardinal of the day 1 in dates is the ordinal (<c>premier</c>, <c>primero</c>, <c>primo</c>, <c>primeiro</c>).</summary>
    public string? FirstOfMonth { get; init; }

    /// <summary>The word that joins the units and the subunits of a currency amount (<c>and</c>, <c>und</c>).</summary>
    public required string CurrencyJoiner { get; init; }
}
