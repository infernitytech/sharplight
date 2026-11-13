using System.Text.RegularExpressions;

namespace Sharplight.Tokenizers.Bpe;

/// <summary>Code points of a text fragment plus the char offset where each starts (length n+1).</summary>
internal readonly ref struct SplitInput(ReadOnlySpan<char> text, ReadOnlySpan<int> cpts, ReadOnlySpan<int> charStart)
{
    public readonly ReadOnlySpan<char> Text = text;
    public readonly ReadOnlySpan<int> Cpts = cpts;
    public readonly ReadOnlySpan<int> CharStart = charStart;
}

/// <summary>Splits a fragment into "words" (lengths measured in code points, summing to the fragment's length).</summary>
internal abstract class PreTokenizer
{
    public abstract void Split(in SplitInput input, List<int> lengths);
}

/// <summary>A resolved <c>tokenizer.pre</c> value: splitter plus the BPE behaviours llama.cpp ties to it.</summary>
internal sealed record PreTokenizerPreset(PreTokenizer Splitter, bool IgnoreMerges, bool DefaultAddBos)
{
    public static PreTokenizerPreset Resolve(string? name)
    {
        switch (name)
        {
            case null or "" or "default":
                return new(DefaultRegexStages.Instance, false, false);

            case "qwen35":
                return new(new RegexFamilySplitter(RegexFamilySplitter.Kind.Llama, caseInsensitiveContractions: true, maxDigits: 1, marks: true), false, false);

            case "qwen2" or "deepseek-r1-qwen" or "kormo" or "f2llmv2" or "megrez" or "stablelm2" or "hunyuan" or "solar-open":
                return new(new RegexFamilySplitter(RegexFamilySplitter.Kind.Llama, true, 1, false), false, false);

            case "llama3" or "llama-v3" or "llama-bpe" or "falcon3" or "falcon-h1" or "midm-2.0" or "jina-v5-nano":
                return new(new RegexFamilySplitter(RegexFamilySplitter.Kind.Llama, true, 3, false), true, true);

            case "lfm2":
                return new(new RegexFamilySplitter(RegexFamilySplitter.Kind.Llama, true, 3, false), true, false);

            // llama.cpp: digits are split individually first, then the GPT-2 expression (without its trailing "\\s+" alternative).
            case "smollm" or "codeshell" or "exaone" or "minerva":
                return new(new DefaultRegexStages([@"\p{N}", $@"'s|'t|'re|'ve|'m|'ll|'d| ?\p{{L}}+| ?\p{{N}}+| ?[^{DefaultRegexStages.Ws}\p{{L}}\p{{N}}]+|[{DefaultRegexStages.Ws}]+(?![^{DefaultRegexStages.Ws}])"]), false, false);

            case "gpt-2" or "gpt2" or "phi-2" or "jina-es" or "jina-de" or "gigachat" or "jina-v2-es" or "jina-v2-de"
                or "a.x-4.0" or "mellum" or "modern-bert" or "mpt" or "olmo" or "jais" or "trillion" or "granite-docling":
                return new(new RegexFamilySplitter(RegexFamilySplitter.Kind.Gpt2, false, 0, false), false, false);

            default:
                throw new NotSupportedException($"Pre-tokenizer '{name}' is not supported.");
        }
    }
}

/// <summary>
/// Hand-written equivalents of llama.cpp's <c>unicode_regex_split_custom_*</c> functions (GPT-2, Llama-3, Qwen2, Qwen3.5 regexes).
/// Faster than a regex engine and bug-compatible with llama.cpp, including its treatment of unassigned code points.
/// </summary>
internal sealed class RegexFamilySplitter(RegexFamilySplitter.Kind kind, bool caseInsensitiveContractions, int maxDigits, bool marks) : PreTokenizer
{
    public enum Kind { Gpt2, Llama }

    public override void Split(in SplitInput input, List<int> lengths)
    {
        if (kind == Kind.Gpt2) SplitGpt2(input.Cpts, lengths);
        else SplitLlama(input.Cpts, lengths);
    }

    private static byte F(ReadOnlySpan<int> c, int pos) => (uint)pos < (uint)c.Length ? UnicodeInfo.Flags(c[pos]) : (byte)0;

    private static int Lower(int cp) => (uint)(cp - 'A') <= 'Z' - 'A' ? cp + 32 : cp;

    private static void Add(List<int> lengths, ref int prevEnd, int end)
    {
        int len = end - prevEnd;
        if (len > 0) lengths.Add(len);
        prevEnd = end;
    }

    // (?:'s|'t|'re|'ve|'m|'ll|'d), case-insensitive for llama-style; returns match length or 0
    private int Contraction(ReadOnlySpan<int> c, int pos)
    {
        int n = c.Length;
        if (c[pos] != '\'' || pos + 1 >= n) return 0;
        int a = c[pos + 1], b = pos + 2 < n ? c[pos + 2] : -1;
        if (caseInsensitiveContractions) { a = Lower(a); b = Lower(b); }
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
        if (pos + 2 < n && ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l'))) return 3;
        return 0;
    }

    private void SplitGpt2(ReadOnlySpan<int> c, List<int> lengths)
    {
        int n = c.Length, pos = 0, prevEnd = 0;
        while (pos < n)
        {
            int cpt = c[pos];
            byte flags = F(c, pos);

            int cl = Contraction(c, pos);
            if (cl > 0) { pos += cl; Add(lengths, ref prevEnd, pos); continue; }

            byte f2 = cpt == ' ' ? F(c, pos + 1) : flags;
            if ((f2 & UnicodeInfo.Letter) != 0)
            {
                pos += cpt == ' ' ? 1 : 0;
                while ((f2 & UnicodeInfo.Letter) != 0) f2 = F(c, ++pos);
                Add(lengths, ref prevEnd, pos);
                continue;
            }
            if ((f2 & UnicodeInfo.Number) != 0)
            {
                pos += cpt == ' ' ? 1 : 0;
                while ((f2 & UnicodeInfo.Number) != 0) f2 = F(c, ++pos);
                Add(lengths, ref prevEnd, pos);
                continue;
            }
            const byte Excl = UnicodeInfo.Whitespace | UnicodeInfo.Letter | UnicodeInfo.Number;
            if ((f2 & Excl) == 0 && f2 != 0)
            {
                pos += cpt == ' ' ? 1 : 0;
                while ((f2 & Excl) == 0 && f2 != 0) f2 = F(c, ++pos);
                Add(lengths, ref prevEnd, pos);
                continue;
            }

            int nws = 0;
            while ((F(c, pos + nws) & UnicodeInfo.Whitespace) != 0) nws++;
            if (nws > 1 && pos + nws < n) { pos += nws - 1; Add(lengths, ref prevEnd, pos); continue; }
            if (nws > 0) { pos += nws; Add(lengths, ref prevEnd, pos); continue; }
            Add(lengths, ref prevEnd, ++pos);
        }
    }

    private void SplitLlama(ReadOnlySpan<int> c, List<int> lengths)
    {
        int n = c.Length, pos = 0, prevEnd = 0;
        byte letterMask = marks ? (byte)(UnicodeInfo.Letter | UnicodeInfo.Mark) : UnicodeInfo.Letter;
        byte punctExcl = (byte)(UnicodeInfo.Whitespace | UnicodeInfo.Number | letterMask);
        while (pos < n)
        {
            int cpt = c[pos];
            byte flags = F(c, pos);

            int cl = Contraction(c, pos);
            if (cl > 0) { pos += cl; Add(lengths, ref prevEnd, pos); continue; }

            // [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
            if (!(cpt == '\r' || cpt == '\n' || (flags & UnicodeInfo.Number) != 0))
            {
                if ((flags & letterMask) != 0 || (F(c, pos + 1) & letterMask) != 0)
                {
                    pos++;
                    while ((F(c, pos) & letterMask) != 0) pos++;
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
                    if (++pos - ini >= maxDigits) { Add(lengths, ref prevEnd, pos); ini = pos; }
                }
                Add(lengths, ref prevEnd, pos);
                continue;
            }

            // ' '?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
            byte f2 = cpt == ' ' ? F(c, pos + 1) : flags;
            if ((f2 & punctExcl) == 0 && flags != 0)
            {
                pos += cpt == ' ' ? 1 : 0;
                while ((f2 & punctExcl) == 0 && f2 != 0) f2 = F(c, ++pos);
                while (pos < n && (c[pos] == '\r' || c[pos] == '\n')) pos++;
                Add(lengths, ref prevEnd, pos);
                continue;
            }

            int nws = 0, lastEndRn = 0;
            while ((F(c, pos + nws) & UnicodeInfo.Whitespace) != 0)
            {
                int cp2 = c[pos + nws];
                if (cp2 == '\r' || cp2 == '\n') lastEndRn = pos + nws + 1;
                nws++;
            }
            if (lastEndRn > 0) { pos = lastEndRn; Add(lengths, ref prevEnd, pos); continue; }      // \s*[\r\n]+
            if (nws > 1 && pos + nws < n) { pos += nws - 1; Add(lengths, ref prevEnd, pos); continue; } // \s+(?!\S)
            if (nws > 0) { pos += nws; Add(lengths, ref prevEnd, pos); continue; }                    // \s+
            Add(lengths, ref prevEnd, ++pos);
        }
    }
}

/// <summary>
/// llama.cpp's generic "default" BPE pre-tokenizer: several successive regex stages, each splitting the pieces of the previous one
/// (text between matches is kept as its own piece). Uses .NET Regex with llama.cpp's explicit whitespace set instead of \s.
/// </summary>
internal sealed class DefaultRegexStages : PreTokenizer
{
    public DefaultRegexStages() { _stages = DefaultStages; }

    /// <summary>A splitter with its own successive regex stages (same semantics as the default one).</summary>
    public DefaultRegexStages(string[] stagePatterns) { _stages = stagePatterns.Select(New).ToArray(); }

    internal const string Ws = @"\t\n\v\f\r \u0085   -     　";

    private readonly Regex[] _stages;

    private static readonly Regex[] DefaultStages =
    [
        New(@"[\p{P}\$\+<=>\^~\|]+"),
        New($@"'s|'t|'re|'ve|'m|'ll|'d| ?\p{{L}}+| ?\p{{N}}+| ?[^{Ws}\p{{L}}\p{{N}}]+|[{Ws}]+(?![^{Ws}])"),
        New(@"\p{N}+"),
        New(@"[0-9][0-9][0-9]"),
    ];

    public static readonly DefaultRegexStages Instance = new();

    private static Regex New(string p) => new(p, RegexOptions.CultureInvariant | RegexOptions.Compiled);

    public override void Split(in SplitInput input, List<int> lengths)
    {
        var pieces = new List<(int start, int len)> { (0, input.Cpts.Length) };
        foreach (var stage in _stages)
        {
            var next = new List<(int, int)>(pieces.Count);
            foreach (var (start, len) in pieces)
            {
                int cs = input.CharStart[start], ce = input.CharStart[start + len];
                var span = input.Text[cs..ce];
                int prev = start;
                foreach (var m in stage.EnumerateMatches(span))
                {
                    int ms = CptIndex(input.CharStart, cs + m.Index, start, start + len);
                    int me = CptIndex(input.CharStart, cs + m.Index + m.Length, start, start + len);
                    if (ms > prev) next.Add((prev, ms - prev));
                    if (me > ms) next.Add((ms, me - ms));
                    prev = me;
                }
                if (prev < start + len) next.Add((prev, start + len - prev));
            }
            pieces = next;
        }
        foreach (var (_, len) in pieces) lengths.Add(len);
    }

    private static int CptIndex(ReadOnlySpan<int> charStart, int charPos, int lo, int hi)
    {
        int idx = charStart.Slice(lo, hi - lo + 1).BinarySearch(charPos);
        return lo + (idx >= 0 ? idx : ~idx);
    }
}
