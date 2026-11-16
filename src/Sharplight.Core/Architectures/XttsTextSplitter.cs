using System.Text.RegularExpressions;

namespace Sharplight.Architectures;

/// <summary>
/// Splits text into the pieces XTTS speaks one at a time, as Coqui's synthesizer does: sentences (ending in <c>.</c>, <c>!</c>, <c>?</c> or <c>…</c>
/// before whitespace), each wrapped at word boundaries when it exceeds the length limit.
/// </summary>
internal static partial class XttsTextSplitter
{
    /// <summary>The pieces of <paramref name="text"/>, each at most <paramref name="maxLength"/> characters unless a single word is longer.</summary>
    /// <param name="text">The text.</param>
    /// <param name="maxLength">The longest piece.</param>
    public static IReadOnlyList<string> Split(string text, int maxLength)
    {
        ArgumentNullException.ThrowIfNull(text);
        var pieces = new List<string>();
        string current = "";
        foreach (string sentence in SentenceEnd().Split(text.Trim()).Where(s => s.Length > 0))
        {
            if (sentence.Length <= maxLength)
            {
                pieces.Add(sentence);
                continue;
            }

            foreach (string word in sentence.Split(' ', StringSplitOptions.RemoveEmptyEntries))
            {
                if (current.Length > 0 && current.Length + 1 + word.Length > maxLength)
                {
                    pieces.Add(current);
                    current = "";
                }

                current = current.Length == 0 ? word : current + " " + word;
            }

            if (current.Length > 0)
            {
                pieces.Add(current);
                current = "";
            }
        }

        return pieces;
    }

    [GeneratedRegex(@"(?<=[.!?…])\s+")]
    private static partial Regex SentenceEnd();
}
