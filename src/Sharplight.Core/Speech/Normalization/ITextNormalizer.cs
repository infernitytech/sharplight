namespace Sharplight.Speech.Normalization;

/// <summary>
/// Rewrites written text the way it is spoken (numbers, dates, times, currency amounts, percentages, ordinals and abbreviations spelled out in
/// words) before a speech model tokenizes it. Models differ in what they expect; a synthesizer takes the normalizer from
/// <see cref="SpeechSynthesisOptions.TextNormalizer"/> or falls back to its own default (<see cref="TextNormalizers"/>).
/// </summary>
public interface ITextNormalizer
{
    /// <summary>The language the normalizer writes numbers in (ISO 639-1).</summary>
    public string Language { get; }

    /// <summary>The spoken form of <paramref name="text"/>.</summary>
    /// <param name="text">The written text.</param>
    public string Normalize(string text);
}
