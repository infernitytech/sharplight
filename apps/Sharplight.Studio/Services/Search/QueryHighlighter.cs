namespace Sharplight.Studio.Services.Search;

/// <summary>Finds the words of a question in a passage so the result list can highlight them.</summary>
internal static class QueryHighlighter
{
    private static readonly HashSet<string> s_stopWords = new(StringComparer.Ordinal)
    {
        "the", "a", "an", "of", "to", "in", "on", "at", "is", "are", "was", "were", "be", "and", "or", "for", "do", "does", "how", "what", "who",
        "when", "where", "which", "why", "it", "this", "that", "with", "by", "as", "from", "i", "my", "me", "you", "your",
    };

    /// <summary>The passage cut into plain and matching pieces (the pieces joined are the passage).</summary>
    /// <param name="text">The passage.</param>
    /// <param name="query">The question.</param>
    public static IReadOnlyList<TextSegment> Split(string text, string query)
    {
        var terms = Bm25Index.Tokenize(query).Where(word => word.Length >= 2 && !s_stopWords.Contains(word)).ToHashSet(StringComparer.Ordinal);
        var segments = new List<TextSegment>();
        if (terms.Count == 0)
        {
            segments.Add(new TextSegment(text, false));
            return segments;
        }

        int plain = 0;
        int i = 0;
        while (i < text.Length)
        {
            if (!char.IsLetterOrDigit(text[i]))
            {
                i++;
                continue;
            }

            int start = i;
            while (i < text.Length && char.IsLetterOrDigit(text[i]))
            {
                i++;
            }

            if (terms.Contains(text[start..i].ToLowerInvariant()))
            {
                if (start > plain)
                {
                    segments.Add(new TextSegment(text[plain..start], false));
                }

                segments.Add(new TextSegment(text[start..i], true));
                plain = i;
            }
        }

        if (plain < text.Length)
        {
            segments.Add(new TextSegment(text[plain..], false));
        }

        return segments;
    }
}
