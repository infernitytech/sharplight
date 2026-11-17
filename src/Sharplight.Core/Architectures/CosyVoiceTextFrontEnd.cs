using System.Text;

namespace Sharplight.Architectures;

/// <summary>
/// CosyVoice's text front end without its optional normalizers (<c>CosyVoiceFrontEnd.text_normalize</c> when neither ttsfrd nor wetext is installed):
/// digits spelled as inflect spells them for text without Chinese characters, and the paragraph split into utterances at sentence punctuation, merged
/// up to 80 tokens (characters for Chinese text; a last piece under 20 is appended to the previous one). Text with markup (<c>&lt;|…|&gt;</c>) is
/// passed through whole.
/// </summary>
internal static class CosyVoiceTextFrontEnd
{
    private static readonly string[] s_ones =
    [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
    ];

    private static readonly string[] s_tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];

    private static readonly string[] s_scales = ["", "thousand", "million", "billion", "trillion", "quadrillion"];

    /// <summary>The utterances of a text.</summary>
    /// <param name="text">The text.</param>
    /// <param name="tokenCount">Counts the tokens of a piece of text.</param>
    public static IReadOnlyList<string> Split(string text, Func<string, int> tokenCount)
    {
        text = text.Trim();
        if (text.Length == 0)
        {
            return [];
        }

        if (text.Contains("<|", StringComparison.Ordinal) && text.Contains("|>", StringComparison.Ordinal))
        {
            return [text];
        }

        bool chinese = text.Any(IsChinese);
        if (chinese)
        {
            text = text.Replace("\n", "", StringComparison.Ordinal).Replace(".", "。", StringComparison.Ordinal).Replace(" - ", "，", StringComparison.Ordinal);
        }
        else
        {
            text = SpellNumbers(text);
        }

        char[] punctuation = chinese ? ['。', '？', '！', '；', '：', '、', '.', '?', '!', ';'] : ['.', '?', '!', ';', ':'];
        if (Array.IndexOf(punctuation, text[^1]) < 0)
        {
            text += chinese ? "。" : ".";
        }

        int Length(string piece) => chinese ? piece.Length : tokenCount(piece);
        var pieces = new List<string>();
        int start = 0;
        for (int i = 0; i < text.Length; i++)
        {
            if (Array.IndexOf(punctuation, text[i]) < 0)
            {
                continue;
            }

            if (i > start)
            {
                pieces.Add(text[start..(i + 1)]);
            }

            if (i + 1 < text.Length && text[i + 1] is '"' or '”' && pieces.Count > 0)
            {
                pieces[^1] += text[i + 1];
                start = i + 2;
            }
            else
            {
                start = i + 1;
            }
        }

        var result = new List<string>();
        string current = "";
        foreach (string piece in pieces)
        {
            if (Length(current + piece) > 80 && Length(current) > 60)
            {
                result.Add(current);
                current = "";
            }

            current += piece;
        }

        if (current.Length > 0)
        {
            if (Length(current) < 20 && result.Count > 0)
            {
                result[^1] += current;
            }
            else
            {
                result.Add(current);
            }
        }

        return [.. result.Where(r => r.Any(c => char.IsLetterOrDigit(c)))];
    }

    /// <summary>Every run of digits spelled as inflect's <c>number_to_words</c> spells it (<c>1234</c> → one thousand, two hundred and thirty-four).</summary>
    /// <param name="text">The text.</param>
    public static string SpellNumbers(string text)
    {
        var result = new StringBuilder(text.Length);
        int i = 0;
        while (i < text.Length)
        {
            if (!char.IsAsciiDigit(text[i]))
            {
                result.Append(text[i++]);
                continue;
            }

            int start = i;
            while (i < text.Length && char.IsAsciiDigit(text[i]))
            {
                i++;
            }

            result.Append(Spell(text[start..i]));
        }

        return result.ToString();
    }

    // inflect's cardinal: groups of three joined by ", ", "and" before the tens of a group with hundreds and before a last group under a hundred.
    private static string Spell(string digits)
    {
        string trimmed = digits.TrimStart('0');
        if (trimmed.Length == 0)
        {
            return "zero";
        }

        if (trimmed.Length > 3 * s_scales.Length)
        {
            return string.Join(' ', trimmed.Select(c => s_ones[c - '0']));
        }

        var groups = new List<int>();
        for (int end = trimmed.Length; end > 0; end -= 3)
        {
            groups.Add(int.Parse(trimmed[Math.Max(0, end - 3)..end], System.Globalization.CultureInfo.InvariantCulture));
        }

        var parts = new List<string>();
        for (int g = groups.Count - 1; g >= 0; g--)
        {
            if (groups[g] == 0)
            {
                continue;
            }

            string words = Group(groups[g]);
            if (g > 0)
            {
                words += " " + s_scales[g];
            }

            if (g == 0 && groups[0] < 100 && parts.Count > 0)
            {
                parts[^1] += " and " + words;
            }
            else
            {
                parts.Add(words);
            }
        }

        return string.Join(", ", parts);
    }

    private static string Group(int n)
    {
        int hundreds = n / 100;
        int rest = n % 100;
        string tens = rest < 20 ? s_ones[rest] : s_tens[rest / 10] + (rest % 10 > 0 ? "-" + s_ones[rest % 10] : "");
        if (hundreds == 0)
        {
            return tens;
        }

        return rest == 0 ? $"{s_ones[hundreds]} hundred" : $"{s_ones[hundreds]} hundred and {tens}";
    }

    private static bool IsChinese(char c) => c is >= '一' and <= '鿿';
}
