using Sharplight.Speech.Normalization;

namespace Sharplight.Speech;

/// <summary>Applies the text normalizer of a request.</summary>
internal static class SpeechTextNormalization
{
    extension(SpeechSynthesisOptions? options)
    {
        /// <summary>
        /// The text as the model should read it: normalized by the request's <see cref="SpeechSynthesisOptions.TextNormalizer"/>, else by the built-in
        /// normalizer of the request's language (the text's script, else <paramref name="defaultLanguage"/>), unless the request turns normalization off.
        /// </summary>
        /// <param name="text">The text as written.</param>
        /// <param name="defaultLanguage">The language of text without script hints and without a requested language.</param>
        /// <param name="byDefault">Whether the model normalizes without being asked (models with a text front end of their own only when a normalizer is set).</param>
        public string NormalizedText(string text, string defaultLanguage = "en", bool byDefault = true)
        {
            if (options?.TextNormalizer is null && (!byDefault || options is { NormalizeText: false }))
            {
                return text;
            }

            if (options?.TextNormalizer is { } custom)
            {
                return custom.Normalize(text);
            }

            string language = options?.Language is { Length: > 0 } requested ? requested : TextNormalizers.Detect(text, defaultLanguage);
            return TextNormalizers.For(language).Normalize(text);
        }
    }
}
