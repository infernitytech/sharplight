using System.Globalization;
using System.Text;

namespace Sharplight.Architectures;

/// <summary>
/// Turns text into the phoneme symbol chunks the Kokoro graphs run on: English words through the model's pronunciation lexicons (with plural, past tense,
/// progressive and possessive endings), unknown words through letter-to-sound rules, numbers read as words, punctuation passed through. This follows the
/// structure of the misaki front end Kokoro was trained with but not its part-of-speech dependent choices, so it is an approximation of it.
/// </summary>
internal sealed class KokoroPhonemizer
{
    /// <summary>The <c>Language</c> option value that marks the text as phonemes already (the symbols of the model's vocabulary).</summary>
    public const string PhonemeLanguage = "phonemes";

    private const string Punctuation = ";:,.!?—…\"“”()";
    private const string Vowels = "AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ";
    private readonly KokoroLexicon _primary;
    private readonly KokoroLexicon? _secondary;

    /// <summary>Creates a phonemizer over lexicons.</summary>
    /// <param name="primary">The lexicon consulted first (the gold dictionary).</param>
    /// <param name="secondary">The lexicon consulted when the first has no entry (the silver dictionary), or <see langword="null"/>.</param>
    public KokoroPhonemizer(KokoroLexicon primary, KokoroLexicon? secondary)
    {
        _primary = primary;
        _secondary = secondary;
    }

    /// <summary>Splits <paramref name="text"/> into one chunk per sentence (longer sentences are split at word boundaries into chunks of at most <paramref name="maxSymbols"/> phoneme symbols).</summary>
    /// <param name="text">The text, or phonemes when <paramref name="language"/> is <see cref="PhonemeLanguage"/>.</param>
    /// <param name="language">The language code or <see cref="PhonemeLanguage"/>.</param>
    /// <param name="maxSymbols">The longest chunk.</param>
    public IEnumerable<string> Chunks(string text, string? language, int maxSymbols)
    {
        if (language == PhonemeLanguage)
        {
            return Split(text, maxSymbols);
        }

        var sentences = new List<string>();
        foreach (string sentence in Sentences(text))
        {
            string phonemes = Phonemize(sentence);
            if (phonemes.Length > 0)
            {
                sentences.Add(phonemes);
            }
        }

        return sentences.SelectMany(sentence => Split(sentence, maxSymbols));
    }

    /// <summary>The phonemes of one passage.</summary>
    /// <param name="text">The text.</param>
    public string Phonemize(string text)
    {
        var builder = new StringBuilder();
        int i = 0;
        string normalized = text.Normalize(NormalizationForm.FormC).Replace('’', '\'').Replace('‘', '\'');
        while (i < normalized.Length)
        {
            char c = normalized[i];
            if (char.IsWhiteSpace(c))
            {
                if (builder.Length > 0 && builder[^1] != ' ')
                {
                    builder.Append(' ');
                }

                i++;
            }
            else if (char.IsLetter(c))
            {
                int start = i;
                while (i < normalized.Length && (char.IsLetter(normalized[i]) || (normalized[i] == '\'' && i + 1 < normalized.Length && char.IsLetter(normalized[i + 1]))))
                {
                    i++;
                }

                AppendWord(builder, normalized[start..i]);
            }
            else if (char.IsDigit(c))
            {
                int start = i;
                while (i < normalized.Length && char.IsDigit(normalized[i]))
                {
                    i++;
                }

                foreach (string word in NumberWords(normalized[start..i]))
                {
                    AppendWord(builder, word);
                }
            }
            else
            {
                if (Punctuation.Contains(c, StringComparison.Ordinal))
                {
                    while (builder.Length > 0 && builder[^1] == ' ')
                    {
                        builder.Length--;
                    }

                    builder.Append(c);
                }
                else if (c is '%' or '&' or '+' or '@')
                {
                    AppendWord(builder, c switch { '%' => "percent", '&' => "and", '+' => "plus", _ => "at" });
                }

                i++;
            }
        }

        return builder.ToString().Trim();
    }

    private void AppendWord(StringBuilder builder, string word)
    {
        if (builder.Length > 0 && builder[^1] != ' ')
        {
            builder.Append(' ');
        }

        string phonemes = Pronounce(word);
        if (word.Equals("a", StringComparison.OrdinalIgnoreCase) && phonemes == "A")
        {
            phonemes = "ɐ";
        }

        // The article is reduced before a consonant and kept long before a vowel.
        if (builder.Length >= 3 && builder.ToString(builder.Length - 3, 3) == "ðə " && Vowels.Contains(FirstPhoneme(phonemes), StringComparison.Ordinal))
        {
            builder[^2] = 'i';
        }

        builder.Append(word.Equals("the", StringComparison.OrdinalIgnoreCase) ? "ðə" : phonemes);
    }

    private static char FirstPhoneme(string phonemes) => phonemes.FirstOrDefault(c => c is not ('ˈ' or 'ˌ'));

    private string Pronounce(string word)
    {
        string? found = Lookup(word);
        if (found is not null)
        {
            return found;
        }

        string lower = word.ToLowerInvariant();
        if (lower.EndsWith("'s", StringComparison.Ordinal) && Lookup(word[..^2]) is { } possessive)
        {
            return possessive + SibilantSuffix(possessive);
        }

        if (lower.EndsWith('s') && lower.Length > 3 && !lower.EndsWith("ss", StringComparison.Ordinal))
        {
            string stem = word[..^1];
            if (lower.EndsWith("es", StringComparison.Ordinal) && Lookup(word[..^2]) is { } esStem && "sz".Contains(lower[^3]))
            {
                return esStem + "ɪz";
            }

            if (Lookup(stem) is { } plural)
            {
                return plural + SibilantSuffix(plural);
            }
        }

        if (lower.EndsWith("ed", StringComparison.Ordinal) && lower.Length > 4)
        {
            foreach (string stem in new[] { word[..^2], word[..^1], DropDouble(word[..^2]) })
            {
                if (Lookup(stem) is { } past)
                {
                    return past + PastSuffix(past);
                }
            }
        }

        if (lower.EndsWith("ing", StringComparison.Ordinal) && lower.Length > 5)
        {
            foreach (string stem in new[] { word[..^3], word[..^3] + "e", DropDouble(word[..^3]) })
            {
                if (Lookup(stem) is { } progressive)
                {
                    return progressive + "ɪŋ";
                }
            }
        }

        return LetterToSound(lower);
    }

    private string? Lookup(string word) => _primary.Find(word) ?? _secondary?.Find(word);

    private static string DropDouble(string stem) => stem.Length > 2 && stem[^1] == stem[^2] ? stem[..^1] : stem;

    private static char LastPhoneme(string phonemes)
    {
        for (int i = phonemes.Length - 1; i >= 0; i--)
        {
            if (phonemes[i] is not ('ˈ' or 'ˌ'))
            {
                return phonemes[i];
            }
        }

        return ' ';
    }

    private static string SibilantSuffix(string phonemes)
    {
        char last = LastPhoneme(phonemes);
        return "sʃzʒʧʤ".Contains(last, StringComparison.Ordinal) ? "ɪz" : "ptkfθ".Contains(last, StringComparison.Ordinal) ? "s" : "z";
    }

    private static string PastSuffix(string phonemes)
    {
        char last = LastPhoneme(phonemes);
        return last is 't' or 'd' ? "ɪd" : "pkfθʃsʧ".Contains(last, StringComparison.Ordinal) ? "t" : "d";
    }

    /// <summary>
    /// Rule-based pronunciation of a word the lexicons lack: digraphs, soft c and g, a final silent e lengthening the vowel before it, stress on the first vowel.
    /// </summary>
    private static string LetterToSound(string word)
    {
        var output = new StringBuilder();
        bool stressed = false;
        int i = 0;
        while (i < word.Length)
        {
            string rest = word[i..];
            bool atEnd = i == word.Length - 1;
            (string Phonemes, int Length)? rule = rest switch
            {
                _ when rest.StartsWith("tch", StringComparison.Ordinal) => ("ʧ", 3),
                _ when rest.StartsWith("th", StringComparison.Ordinal) => ("θ", 2),
                _ when rest.StartsWith("sh", StringComparison.Ordinal) => ("ʃ", 2),
                _ when rest.StartsWith("ch", StringComparison.Ordinal) => ("ʧ", 2),
                _ when rest.StartsWith("ph", StringComparison.Ordinal) => ("f", 2),
                _ when rest.StartsWith("ck", StringComparison.Ordinal) => ("k", 2),
                _ when rest.StartsWith("ng", StringComparison.Ordinal) => ("ŋ", 2),
                _ when rest.StartsWith("wh", StringComparison.Ordinal) => ("w", 2),
                _ when rest.StartsWith("qu", StringComparison.Ordinal) => ("kw", 2),
                _ when rest.StartsWith("ee", StringComparison.Ordinal) || rest.StartsWith("ea", StringComparison.Ordinal) => ("i", 2),
                _ when rest.StartsWith("oo", StringComparison.Ordinal) => ("u", 2),
                _ when rest.StartsWith("ai", StringComparison.Ordinal) || rest.StartsWith("ay", StringComparison.Ordinal) => ("A", 2),
                _ when rest.StartsWith("oa", StringComparison.Ordinal) || rest.StartsWith("ow", StringComparison.Ordinal) => ("O", 2),
                _ when rest.StartsWith("ou", StringComparison.Ordinal) => ("W", 2),
                _ when rest.StartsWith("oi", StringComparison.Ordinal) || rest.StartsWith("oy", StringComparison.Ordinal) => ("Y", 2),
                _ when rest.StartsWith("ar", StringComparison.Ordinal) => ("ɑɹ", 2),
                _ when rest.StartsWith("or", StringComparison.Ordinal) => ("ɔɹ", 2),
                _ when rest.StartsWith("er", StringComparison.Ordinal) || rest.StartsWith("ir", StringComparison.Ordinal) || rest.StartsWith("ur", StringComparison.Ordinal) => ("ɜɹ", 2),
                _ => null,
            };
            string phonemes;
            int length;
            if (rule is { } r)
            {
                (phonemes, length) = r;
            }
            else
            {
                char c = word[i];
                char next = i + 1 < word.Length ? word[i + 1] : '\0';
                bool magicE = i + 2 < word.Length && word[i + 2] == 'e' && i + 3 == word.Length && "bcdfgklmnprstvz".Contains(next, StringComparison.Ordinal);
                length = 1;
                phonemes = c switch
                {
                    'a' => magicE ? "A" : "æ",
                    'e' => atEnd && i > 0 ? string.Empty : "ɛ",
                    'i' => magicE ? "I" : "ɪ",
                    'o' => magicE ? "O" : "ɑ",
                    'u' => magicE ? "ju" : "ʌ",
                    'y' => i == 0 ? "j" : atEnd ? "i" : "ɪ",
                    'c' => "eiy".Contains(next, StringComparison.Ordinal) ? "s" : "k",
                    'g' => "ei".Contains(next, StringComparison.Ordinal) ? "ʤ" : "ɡ",
                    'j' => "ʤ",
                    'q' => "k",
                    'x' => "ks",
                    'r' => "ɹ",
                    _ when char.IsLetter(c) && c < 128 => c.ToString(),
                    _ => string.Empty,
                };
                if (i + 1 < word.Length && word[i + 1] == c && !"aeiou".Contains(c, StringComparison.Ordinal))
                {
                    length = 2;
                }
            }

            if (!stressed && phonemes.Length > 0 && Vowels.Contains(phonemes[0], StringComparison.Ordinal))
            {
                output.Append('ˈ');
                stressed = true;
            }

            output.Append(phonemes);
            i += length;
        }

        return output.ToString();
    }

    private static IEnumerable<string> NumberWords(string digits)
    {
        if (!long.TryParse(digits, NumberStyles.None, CultureInfo.InvariantCulture, out long value) || digits.Length > 15)
        {
            return digits.Select(d => d switch { '0' => "zero", '1' => "one", '2' => "two", '3' => "three", '4' => "four", '5' => "five", '6' => "six", '7' => "seven", '8' => "eight", _ => "nine" });
        }

        return Spell(value).Split(' ', StringSplitOptions.RemoveEmptyEntries);
    }

    private static string Spell(long n)
    {
        string[] small = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
            "sixteen", "seventeen", "eighteen", "nineteen"];
        string[] tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];
        if (n < 20)
        {
            return small[n];
        }

        if (n < 100)
        {
            return tens[n / 10] + (n % 10 == 0 ? string.Empty : " " + small[n % 10]);
        }

        if (n < 1000)
        {
            return small[n / 100] + " hundred" + (n % 100 == 0 ? string.Empty : " " + Spell(n % 100));
        }

        (long scale, string name)[] scales = [(1_000_000_000_000, "trillion"), (1_000_000_000, "billion"), (1_000_000, "million"), (1000, "thousand")];
        foreach (var (scale, name) in scales)
        {
            if (n >= scale)
            {
                return Spell(n / scale) + " " + name + (n % scale == 0 ? string.Empty : " " + Spell(n % scale));
            }
        }

        return small[0];
    }

    private static IEnumerable<string> Sentences(string text)
    {
        int start = 0;
        for (int i = 0; i < text.Length; i++)
        {
            if (text[i] is '.' or '!' or '?' or '…' && (i + 1 == text.Length || char.IsWhiteSpace(text[i + 1])))
            {
                yield return text[start..(i + 1)];
                start = i + 1;
            }
        }

        if (start < text.Length)
        {
            yield return text[start..];
        }
    }

    private static IEnumerable<string> Split(string phonemes, int maxSymbols)
    {
        int start = 0;
        while (start < phonemes.Length)
        {
            int length = Math.Min(maxSymbols, phonemes.Length - start);
            if (start + length < phonemes.Length)
            {
                int space = phonemes.LastIndexOf(' ', start + length - 1, length);
                if (space > start)
                {
                    length = space - start;
                }
            }

            yield return phonemes.Substring(start, length).Trim();
            start += length;
        }
    }
}
