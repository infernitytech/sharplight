namespace Sharplight.Tokenizers.Bpe;

/// <summary>
/// Hand-written equivalents of llama.cpp's <c>unicode_regex_split_custom_*</c> functions (GPT-2, Llama-3, Qwen2, Qwen3.5 regexes).
/// Faster than a regex engine and bug-compatible with llama.cpp, including its treatment of unassigned code points.
/// </summary>
internal sealed class RegexFamilySplitter(RegexFamily family, bool caseInsensitiveContractions, int maxDigits, bool marks) : PreTokenizer
{
    public override void Split(in SplitInput input, List<int> lengths)
    {
        if (family == RegexFamily.Gpt2)
        {
            SplitGpt2(input.Cpts, lengths);
        }
        else
        {
            SplitLlama(input.Cpts, lengths);
        }
    }

    private static byte F(ReadOnlySpan<int> c, int pos) => (uint)pos < (uint)c.Length ? UnicodeInfo.Flags(c[pos]) : (byte)0;

    private static int Lower(int cp) => (uint)(cp - 'A') <= 'Z' - 'A' ? cp + 32 : cp;

    private static void Add(List<int> lengths, ref int prevEnd, int end)
    {
        int len = end - prevEnd;
        if (len > 0)
        {
            lengths.Add(len);
        }

        prevEnd = end;
    }

    // (?:'s|'t|'re|'ve|'m|'ll|'d), case-insensitive for llama-style; returns match length or 0
    private int Contraction(ReadOnlySpan<int> c, int pos)
    {
        int n = c.Length;
        if (c[pos] != '\'' || pos + 1 >= n)
        {
            return 0;
        }

        int a = c[pos + 1];
        int b = pos + 2 < n ? c[pos + 2] : -1;
        if (caseInsensitiveContractions)
        {
            a = Lower(a);
            b = Lower(b);
        }
        if (a == 's' || a == 't' || a == 'm' || a == 'd')
        {
            return 2;
        }

        if (pos + 2 < n && ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')))
        {
            return 3;
        }

        return 0;
    }

    private void SplitGpt2(ReadOnlySpan<int> c, List<int> lengths)
    {
        int n = c.Length;
        int pos = 0;
        int prevEnd = 0;
        while (pos < n)
        {
            int cpt = c[pos];
            byte flags = F(c, pos);

            int cl = Contraction(c, pos);
            if (cl > 0)
            {
                pos += cl;
                Add(lengths, ref prevEnd, pos);
                continue;
            }

            byte f2 = cpt == ' ' ? F(c, pos + 1) : flags;
            if ((f2 & UnicodeInfo.Letter) != 0)
            {
                pos += cpt == ' ' ? 1 : 0;
                while ((f2 & UnicodeInfo.Letter) != 0)
                {
                    f2 = F(c, ++pos);
                }

                Add(lengths, ref prevEnd, pos);
                continue;
            }
            if ((f2 & UnicodeInfo.Number) != 0)
            {
                pos += cpt == ' ' ? 1 : 0;
                while ((f2 & UnicodeInfo.Number) != 0)
                {
                    f2 = F(c, ++pos);
                }

                Add(lengths, ref prevEnd, pos);
                continue;
            }
            const byte Excl = UnicodeInfo.Whitespace | UnicodeInfo.Letter | UnicodeInfo.Number;
            if ((f2 & Excl) == 0 && f2 != 0)
            {
                pos += cpt == ' ' ? 1 : 0;
                while ((f2 & Excl) == 0 && f2 != 0)
                {
                    f2 = F(c, ++pos);
                }

                Add(lengths, ref prevEnd, pos);
                continue;
            }

            int nws = 0;
            while ((F(c, pos + nws) & UnicodeInfo.Whitespace) != 0)
            {
                nws++;
            }

            if (nws > 1 && pos + nws < n)
            {
                pos += nws - 1;
                Add(lengths, ref prevEnd, pos);
                continue;
            }
            if (nws > 0)
            {
                pos += nws;
                Add(lengths, ref prevEnd, pos);
                continue;
            }
            Add(lengths, ref prevEnd, ++pos);
        }
    }

    private void SplitLlama(ReadOnlySpan<int> c, List<int> lengths)
    {
        int n = c.Length;
        int pos = 0;
        int prevEnd = 0;
        byte letterMask = marks ? (byte)(UnicodeInfo.Letter | UnicodeInfo.Mark) : UnicodeInfo.Letter;
        byte punctExcl = (byte)(UnicodeInfo.Whitespace | UnicodeInfo.Number | letterMask);
        while (pos < n)
        {
            int cpt = c[pos];
            byte flags = F(c, pos);

            int cl = Contraction(c, pos);
            if (cl > 0)
            {
                pos += cl;
                Add(lengths, ref prevEnd, pos);
                continue;
            }

            // [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
            if (!(cpt == '\r' || cpt == '\n' || (flags & UnicodeInfo.Number) != 0))
            {
                if ((flags & letterMask) != 0 || (F(c, pos + 1) & letterMask) != 0)
                {
                    pos++;
                    while ((F(c, pos) & letterMask) != 0)
                    {
                        pos++;
                    }

                    Add(lengths, ref prevEnd, pos);
                    continue;
                }
            }

            // \p{N}  or  \p{N}{1,3}
            if ((flags & UnicodeInfo.Number) != 0)
            {
                int ini = pos;
                while ((F(c, pos) & UnicodeInfo.Number) != 0)
                {
                    if (++pos - ini >= maxDigits)
                    {
                        Add(lengths, ref prevEnd, pos);
                        ini = pos;
                    }
                }
                Add(lengths, ref prevEnd, pos);
                continue;
            }

            // ' '?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
            byte f2 = cpt == ' ' ? F(c, pos + 1) : flags;
            if ((f2 & punctExcl) == 0 && flags != 0)
            {
                pos += cpt == ' ' ? 1 : 0;
                while ((f2 & punctExcl) == 0 && f2 != 0)
                {
                    f2 = F(c, ++pos);
                }

                while (pos < n && (c[pos] == '\r' || c[pos] == '\n'))
                {
                    pos++;
                }

                Add(lengths, ref prevEnd, pos);
                continue;
            }

            int nws = 0;
            int lastEndRn = 0;
            while ((F(c, pos + nws) & UnicodeInfo.Whitespace) != 0)
            {
                int cp2 = c[pos + nws];
                if (cp2 == '\r' || cp2 == '\n')
                {
                    lastEndRn = pos + nws + 1;
                }

                nws++;
            }
            if (lastEndRn > 0)
            {
                pos = lastEndRn;
                Add(lengths, ref prevEnd, pos);
                continue;
            }
            if (nws > 1 && pos + nws < n)
            {
                pos += nws - 1;
                Add(lengths, ref prevEnd, pos);
                continue;
            }
            if (nws > 0)
            {
                pos += nws;
                Add(lengths, ref prevEnd, pos);
                continue;
            }
            Add(lengths, ref prevEnd, ++pos);
        }
    }
}
