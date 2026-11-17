namespace Sharplight.Speech.Normalization;

/// <summary>The names of a currency and its subunit.</summary>
/// <param name="One">The unit, singular.</param>
/// <param name="Many">The unit, plural.</param>
/// <param name="SubOne">The subunit, singular.</param>
/// <param name="SubMany">The subunit, plural.</param>
internal sealed record CurrencyNames(string One, string Many, string SubOne, string SubMany);
