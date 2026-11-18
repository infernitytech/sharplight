namespace Sharplight.Studio.ViewModels.Search;

/// <summary>Cuts a document's text into overlapping passages for embedding, at paragraph, sentence or word ends where it can.</summary>
internal static class TextChunker
{
    /// <summary>The passages of <paramref name="text"/>.</summary>
    /// <param name="text">The document's text.</param>
    /// <param name="size">The most characters in a passage (at least 100).</param>
    /// <param name="overlap">How many characters of a passage the next one repeats (below half of <paramref name="size"/>).</param>
    public static IReadOnlyList<TextChunk> Split(string text, int size, int overlap)
    {
        size = Math.Max(100, size);
        overlap = Math.Clamp(overlap, 0, size / 2);
        var chunks = new List<TextChunk>();
        int position = 0;
        while (position < text.Length)
        {
            int end = Math.Min(text.Length, position + size);
            if (end < text.Length)
            {
                end = BreakBefore(text, position, end);
            }

            var passage = text[position..end].Trim();
            if (passage.Length > 0)
            {
                chunks.Add(new TextChunk(position, passage));
            }

            if (end >= text.Length)
            {
                break;
            }

            position = Math.Max(position + 1, end - overlap);
        }

        return chunks;
    }

    /// <summary>The best place to end a passage in the last fifth of <c>[start, end)</c>: after a paragraph, a sentence or a word.</summary>
    private static int BreakBefore(string text, int start, int end)
    {
        int floor = end - (end - start) / 5;
        for (int i = end - 1; i > floor; i--)
        {
            if (text[i] == '\n' && text[i - 1] == '\n')
            {
                return i + 1;
            }
        }

        for (int i = end - 1; i > floor; i--)
        {
            if (text[i] is '.' or '!' or '?' && i + 1 < text.Length && char.IsWhiteSpace(text[i + 1]))
            {
                return i + 1;
            }
        }

        for (int i = end - 1; i > floor; i--)
        {
            if (char.IsWhiteSpace(text[i]))
            {
                return i + 1;
            }
        }

        return end;
    }
}
