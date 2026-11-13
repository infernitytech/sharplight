namespace Sharplight.Tokenizers;

/// <summary>Splits text around special (control / user-defined) token strings, longest first.</summary>
internal sealed class SpecialTokens
{
    private readonly (string Text, int Id, bool Control)[] _specials;

    public SpecialTokens(string[] tokens, Func<int, TokenType> type)
    {
        _specials = Enumerable.Range(0, tokens.Length)
            .Where(i => type(i) is TokenType.Control or TokenType.UserDefined or TokenType.Unknown && tokens[i].Length > 0)
            .Select(i => (tokens[i], i, type(i) is TokenType.Control or TokenType.Unknown))
            .OrderByDescending(s => s.Item1.Length).ToArray();
    }

    /// <summary>Text fragments (Id = -1) and special tokens in order. Control tokens are only matched with <paramref name="parseSpecial"/>.</summary>
    public List<(string Text, int Id)> Split(string text, bool parseSpecial)
    {
        var buffer = new List<(string, int)>();
        if (text.Length > 0)
        {
            buffer.Add((text, -1));
        }

        foreach (var (stext, sid, control) in _specials)
        {
            if (!parseSpecial && control)
            {
                continue;
            }

            var next = new List<(string, int)>(buffer.Count + 2);
            foreach (var frag in buffer)
            {
                if (frag.Item2 >= 0 || !frag.Item1.Contains(stext, StringComparison.Ordinal)) { next.Add(frag); continue; }
                string rest = frag.Item1;
                while (true)
                {
                    int at = rest.IndexOf(stext, StringComparison.Ordinal);
                    if (at < 0) { if (rest.Length > 0) { next.Add((rest, -1)); } break; }
                    if (at > 0)
                    {
                        next.Add((rest[..at], -1));
                    }

                    next.Add(("", sid));
                    rest = rest[(at + stext.Length)..];
                }
            }
            buffer = next;
        }
        return buffer;
    }
}
