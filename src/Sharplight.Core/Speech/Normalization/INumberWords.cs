namespace Sharplight.Speech.Normalization;

/// <summary>The words of one language's numbers.</summary>
internal interface INumberWords
{
    /// <summary>The cardinal of a non-negative whole number.</summary>
    /// <param name="number">The number.</param>
    public string Cardinal(long number);

    /// <summary>The ordinal (masculine singular) of a positive whole number.</summary>
    /// <param name="number">The number.</param>
    public string Ordinal(long number);

    /// <summary>The way a year is read (<c>1999</c>).</summary>
    /// <param name="year">The year.</param>
    public string Year(long year);

    /// <summary>The word of a single digit, read after a decimal separator or digit by digit.</summary>
    /// <param name="digit">0 to 9.</param>
    public string Digit(int digit);
}
