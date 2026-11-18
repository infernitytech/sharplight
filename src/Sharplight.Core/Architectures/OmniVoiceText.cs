
namespace Sharplight.Architectures;

/// <summary>OmniVoice's text handling as the official package does it: chunking long text at punctuation (avoiding abbreviations), ending a reference transcript with punctuation, and the cleanup of the text that goes into the prompt.</summary>
internal static class OmniVoiceText
{
    private const string SplitPunctuation = ".,;:!?。，；：！？";
    private const string ClosingMarks = "\"'“”‘’）]》>」】";

    private static readonly HashSet<string> s_endPunctuation =
    [
        ";", ":", ",", ".", "!", "?", "…", ")", "]", "}", "\"", "'", "“", "”", "‘", "’", "；", "：", "，", "。", "！", "？", "、", "……", "）", "】",
    ];

    private static readonly HashSet<string> s_abbreviations = new(StringComparer.Ordinal)
    {
        "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "Rev.", "Fr.", "Hon.", "Pres.", "Gov.", "Capt.", "Gen.", "Sen.", "Rep.", "Col.", "Maj.", "Lt.", "Cmdr.", "Sgt.", "Cpl.",
        "Co.", "Corp.", "Inc.", "Ltd.", "Est.", "Dept.", "St.", "Ave.", "Blvd.", "Rd.", "Mt.", "Ft.", "No.", "Jan.", "Feb.", "Mar.", "Apr.", "Aug.", "Sep.", "Sept.", "Oct.", "Nov.",
        "Dec.", "i.e.", "e.g.", "vs.", "Vs.", "Etc.", "approx.", "fig.", "def.",
    };

    /// <summary>Appends a full stop (an ideographic one after Chinese text) unless the text already ends in punctuation.</summary>
    /// <param name="text">The text.</param>
    public static string AddPunctuation(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        text = text.Trim();
        if (text.Length == 0)
        {
            return text;
        }

        string last = char.IsLowSurrogate(text[^1]) && text.Length > 1 ? text[^2..] : text[^1..];
        if (s_endPunctuation.Contains(last) || text.EndsWith("……", StringComparison.Ordinal))
        {
            return text;
        }

        return text + (text.Any(IsIdeograph) ? "。" : ".");
    }

    /// <summary>The prompt's text: the reference transcript and the target joined, line breaks removed, full-width parentheses and runs of blanks normalized, and blanks around ideographs dropped.</summary>
    /// <param name="text">The target text.</param>
    /// <param name="referenceText">The reference transcript, or <see langword="null"/>.</param>
    public static string Combine(string text, string? referenceText)
    {
        string full = string.IsNullOrEmpty(referenceText) ? text.Trim() : referenceText.Trim() + " " + text.Trim();
        full = System.Text.RegularExpressions.Regex.Replace(full, "[\r\n]+", "");
        full = full.Replace('（', '(').Replace('）', ')');
        full = System.Text.RegularExpressions.Regex.Replace(full, "[ \t]+", " ");
        return System.Text.RegularExpressions.Regex.Replace(full, @"(?<=[一-鿿])\s+|\s+(?=[一-鿿])", "");
    }

    /// <summary>Splits text into chunks of about <paramref name="chunkLength"/> characters at punctuation, merging short pieces.</summary>
    /// <param name="text">The text.</param>
    /// <param name="chunkLength">The target length in characters.</param>
    /// <param name="minimumChunkLength">Chunks shorter than this are merged with a neighbor (0 for none).</param>
    public static List<string> Chunk(string text, int chunkLength, int minimumChunkLength = 0)
    {
        ArgumentNullException.ThrowIfNull(text);
        var characters = text.EnumerateRunes().Select(r => r.ToString()).ToList();
        var sentences = new List<List<string>>();
        var current = new List<string>();
        foreach (string token in characters)
        {
            bool punctuation = token.Length == 1 && (SplitPunctuation.Contains(token[0]) || ClosingMarks.Contains(token[0]));
            if (current.Count == 0 && sentences.Count != 0 && punctuation)
            {
                sentences[^1].Add(token);
                continue;
            }

            current.Add(token);
            if (token.Length == 1 && SplitPunctuation.Contains(token[0]))
            {
                bool abbreviation = false;
                if (token == ".")
                {
                    string joined = string.Concat(current).Trim();
                    if (joined.Length > 0)
                    {
                        string lastWord = joined.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries)[^1];
                        abbreviation = s_abbreviations.Contains(lastWord);
                    }
                }

                if (!abbreviation)
                {
                    sentences.Add(current);
                    current = [];
                }
            }
        }

        if (current.Count != 0)
        {
            sentences.Add(current);
        }

        var merged = new List<List<string>>();
        var chunk = new List<string>();
        foreach (var sentence in sentences)
        {
            if (chunk.Count + sentence.Count <= chunkLength)
            {
                chunk.AddRange(sentence);
            }
            else
            {
                if (chunk.Count > 0)
                {
                    merged.Add(chunk);
                }

                chunk = sentence;
            }
        }

        if (chunk.Count > 0)
        {
            merged.Add(chunk);
        }

        var final = new List<List<string>>();
        if (minimumChunkLength > 0)
        {
            bool firstShort = merged.Count > 0 && merged[0].Count < minimumChunkLength;
            for (int i = 0; i < merged.Count; i++)
            {
                if (i == 1 && firstShort)
                {
                    final[^1].AddRange(merged[i]);
                }
                else if (merged[i].Count >= minimumChunkLength || final.Count == 0)
                {
                    final.Add(merged[i]);
                }
                else
                {
                    final[^1].AddRange(merged[i]);
                }
            }
        }
        else
        {
            final = merged;
        }

        return [.. final.Select(c => string.Concat(c).Trim()).Where(s => s.Length > 0)];
    }

    private static bool IsIdeograph(char c) => c is >= '一' and <= '鿿';
}
