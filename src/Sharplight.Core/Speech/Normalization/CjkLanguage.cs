namespace Sharplight.Speech.Normalization;

/// <summary>The words and conventions one Han-script language's <see cref="CjkTextNormalizer"/> needs.</summary>
internal sealed class CjkLanguage
{
    /// <summary>ISO 639-1 code.</summary>
    public required string Code { get; init; }

    /// <summary>The number words.</summary>
    public required INumberWords Words { get; init; }

    /// <summary>Years before 年 are read digit by digit (Chinese) rather than as numbers.</summary>
    public required bool YearByDigits { get; init; }

    /// <summary>The word before a percentage (<c>百分之</c>), or empty.</summary>
    public required string PercentPrefix { get; init; }

    /// <summary>The word after a percentage (<c>パーセント</c>), or empty.</summary>
    public required string PercentSuffix { get; init; }

    /// <summary>The word for a minus sign.</summary>
    public required string Minus { get; init; }

    /// <summary>The word for the decimal point.</summary>
    public required string DecimalWord { get; init; }

    /// <summary>The unit of the hour of a clock time (<c>点</c>, <c>時</c>).</summary>
    public required string HourUnit { get; init; }

    /// <summary>The unit of the minutes of a clock time.</summary>
    public required string MinuteUnit { get; init; }

    /// <summary>The currency symbols and the unit word read after the amount.</summary>
    public required IReadOnlyDictionary<char, string> Currencies { get; init; }

    /// <summary>Symbols with the words read for them.</summary>
    public required (string Symbol, string Word)[] Symbols { get; init; }

    /// <summary>The Chinese 2 before measure words is read 两.</summary>
    public bool TwoBeforeMeasures { get; init; }
}
