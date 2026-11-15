using System.Text;

namespace Sharplight.Tokenizers;

/// <summary>Splits text around special (control / user-defined) token strings, longest first.</summary>
internal sealed class SpecialTokens
{
    private readonly (string Text, int Id, bool Control)[] _specials;

    /// <param name="tokens">The vocabulary.</param>
    /// <param name="type">Type of a token id.</param>
    /// <param name="orderByUtf8Bytes">Rank "longest" by UTF-8 byte count (llama.cpp's SentencePiece and BPE behaviour) instead of by chars.</param>
    public SpecialTokens(string[] tokens, Func<int, TokenType> type, bool orderByUtf8Bytes = false)
    {
        int Length(string text) => orderByUtf8Bytes ? Encoding.UTF8.GetByteCount(text) : text.Length;
        _specials = Enumerable.Range(0, tokens.Length)
            .Where(i => type(i) is TokenType.Control or TokenType.UserDefined or TokenType.Unknown && tokens[i].Length > 0)
            .Select(i => (tokens[i], i, type(i) is TokenType.Control or TokenType.Unknown))
            .OrderByDescending(s => Length(s.Item1))
            .ToArray();
    }

    /// <summary>Text fragments (Id = -1) and special tokens in order. Control tokens are only matched with <paramref name="parseSpecial"/>.</summary>
    public List<(string Text, int Id)> Split(string text, bool parseSpecial)
    {
        var buffer = new List<(string, int)>();
        if (text.Length > 0)
        {
            buffer.Add((text, -1));
        }

        foreach (var (specialText, specialId, control) in _specials)
        {
            if (!parseSpecial && control)
            {
                continue;
            }

            var next = new List<(string, int)>(buffer.Count + 2);
            foreach (var fragment in buffer)
            {
                if (fragment.Item2 >= 0 || !fragment.Item1.Contains(specialText, StringComparison.Ordinal))
                {
                    next.Add(fragment);
                    continue;
                }

                SplitFragment(fragment.Item1, specialText, specialId, next);
            }

            buffer = next;
        }

        return buffer;
    }

    private static void SplitFragment(string fragment, string specialText, int specialId, List<(string, int)> output)
    {
        string rest = fragment;
        while (true)
        {
            int at = rest.IndexOf(specialText, StringComparison.Ordinal);
            if (at < 0)
            {
                if (rest.Length > 0)
                {
                    output.Add((rest, -1));
                }

                return;
            }

            if (at > 0)
            {
                output.Add((rest[..at], -1));
            }

            output.Add(("", specialId));
            rest = rest[(at + specialText.Length)..];
        }
    }
}
