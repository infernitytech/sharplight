using System.Text;

namespace Sharplight.Architectures;

/// <summary>Dia's text tokens: the UTF-8 bytes of the text, with the speaker tags <c>[S1]</c> and <c>[S2]</c> as the bytes 1 and 2.</summary>
internal static class DiaText
{
    private const string FirstSpeaker = "[S1]";
    private const string SecondSpeaker = "[S2]";

    /// <summary>The tokens of <paramref name="text"/>.</summary>
    /// <param name="text">The text to speak.</param>
    public static int[] Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        var tokens = new List<int>(text.Length + 8);
        int i = 0;
        while (i < text.Length)
        {
            if (string.CompareOrdinal(text, i, FirstSpeaker, 0, FirstSpeaker.Length) == 0)
            {
                tokens.Add(1);
                i += FirstSpeaker.Length;
            }
            else if (string.CompareOrdinal(text, i, SecondSpeaker, 0, SecondSpeaker.Length) == 0)
            {
                tokens.Add(2);
                i += SecondSpeaker.Length;
            }
            else
            {
                int next = i + 1;
                while (next < text.Length && text[next] != '[')
                {
                    next++;
                }

                foreach (byte b in Encoding.UTF8.GetBytes(text.Substring(i, next - i)))
                {
                    tokens.Add(b);
                }

                i = next;
            }
        }

        return [.. tokens];
    }

    /// <summary>Whether the text opens with a speaker tag.</summary>
    /// <param name="text">The text.</param>
    public static bool StartsWithSpeaker(string text)
    {
        var trimmed = text.AsSpan().TrimStart();
        return trimmed.StartsWith(FirstSpeaker, StringComparison.Ordinal) || trimmed.StartsWith(SecondSpeaker, StringComparison.Ordinal);
    }

    /// <summary>The text with <c>[S1] </c> in front unless it opens with a speaker tag (the model needs the first speaker's tag).</summary>
    /// <param name="text">The text.</param>
    public static string EnsureSpeaker(string text) => StartsWithSpeaker(text) ? text : FirstSpeaker + " " + text;
}
