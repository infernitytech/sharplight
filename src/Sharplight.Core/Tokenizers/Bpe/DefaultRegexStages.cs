using System.Text.RegularExpressions;

namespace Sharplight.Tokenizers.Bpe;

/// <summary>
/// llama.cpp's generic "default" BPE pre-tokenizer: several successive regex stages, each splitting the pieces of the previous one
/// (text between matches is kept as its own piece). Uses .NET Regex with llama.cpp's explicit whitespace set instead of \s.
/// </summary>
internal sealed class DefaultRegexStages : PreTokenizer
{
    public DefaultRegexStages()
    {
        _stages = DefaultStages;
    }

    /// <summary>A splitter with its own successive regex stages (same semantics as the default one).</summary>
    public DefaultRegexStages(string[] stagePatterns)
    {
        _stages = stagePatterns.Select(New).ToArray();
    }

    /// <summary>llama.cpp's explicit whitespace set as the body of a regex character class (spelled with escapes: it includes U+2028/U+2029).</summary>
    internal const string Ws = @"\t\n\v\f\r \u0085\u00A0\u1680\u2000-\u200A\u2028\u2029\u202F\u205F\u3000";

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
                int cs = input.CharStart[start];
                int ce = input.CharStart[start + len];
                var span = input.Text[cs..ce];
                int prev = start;
                foreach (var m in stage.EnumerateMatches(span))
                {
                    int ms = CptIndex(input.CharStart, cs + m.Index, start, start + len);
                    int me = CptIndex(input.CharStart, cs + m.Index + m.Length, start, start + len);
                    if (ms > prev)
                    {
                        next.Add((prev, ms - prev));
                    }

                    if (me > ms)
                    {
                        next.Add((ms, me - ms));
                    }

                    prev = me;
                }
                if (prev < start + len)
                {
                    next.Add((prev, start + len - prev));
                }
            }
            pieces = next;
        }
        foreach (var (_, len) in pieces)
        {
            lengths.Add(len);
        }
    }

    private static int CptIndex(ReadOnlySpan<int> charStart, int charPos, int lo, int hi)
    {
        int idx = charStart.Slice(lo, hi - lo + 1).BinarySearch(charPos);
        return lo + (idx >= 0 ? idx : ~idx);
    }
}
