using System.Collections.Concurrent;

namespace Sharplight.Speech.Normalization;

/// <summary>
/// The text normalizers of the languages Sharplight knows (<see cref="Supported"/>): numbers, dates, clock times, currency amounts, percentages,
/// ordinals and abbreviations spelled out. Models without a normalizer of their own use these (<see cref="SpeechSynthesisOptions.TextNormalizer"/>
/// replaces them per request).
/// </summary>
public static class TextNormalizers
{
    private static readonly ConcurrentDictionary<string, ITextNormalizer> s_cache = new(StringComparer.Ordinal);

    /// <summary>The ISO 639-1 codes of the supported languages.</summary>
    public static IReadOnlyList<string> Supported { get; } = ["en", "de", "fr", "es", "it", "pt", "zh", "ja"];

    /// <summary>The normalizer of a language.</summary>
    /// <param name="language">An ISO 639-1 code, optionally with a region (<c>de-AT</c>); <see langword="null"/> or unknown languages get the English one.</param>
    public static ITextNormalizer For(string? language)
    {
        string code = (language ?? "en").Split('-', '_')[0].ToLowerInvariant();
        if (!Supported.Contains(code))
        {
            code = "en";
        }

        return s_cache.GetOrAdd(code, Create);
    }

    /// <summary>The language of a text by its script: Japanese when it has kana, Chinese when it has Han characters without kana, else <paramref name="fallback"/>.</summary>
    /// <param name="text">The text.</param>
    /// <param name="fallback">The language of text in other scripts (default English).</param>
    public static string Detect(string text, string fallback = "en")
    {
        ArgumentNullException.ThrowIfNull(text);
        bool han = false;
        foreach (char c in text)
        {
            if (c is >= '぀' and <= 'ヿ')
            {
                return "ja";
            }

            han |= c is >= '一' and <= '鿿';
        }

        return han ? "zh" : fallback;
    }

    private static ITextNormalizer Create(string code) => code switch
    {
        "de" => new AlphabeticTextNormalizer(AlphabeticLanguages.German),
        "fr" => new AlphabeticTextNormalizer(AlphabeticLanguages.French),
        "es" => new AlphabeticTextNormalizer(AlphabeticLanguages.Spanish),
        "it" => new AlphabeticTextNormalizer(AlphabeticLanguages.Italian),
        "pt" => new AlphabeticTextNormalizer(AlphabeticLanguages.Portuguese),
        "zh" => new CjkTextNormalizer(CjkLanguages.Chinese),
        "ja" => new CjkTextNormalizer(CjkLanguages.Japanese),
        _ => new AlphabeticTextNormalizer(AlphabeticLanguages.English),
    };
}
