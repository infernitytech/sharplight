using System.Text;

namespace Sharplight.Llm.Structured;

/// <summary>
/// A set of Unicode code points (inclusive ranges), compiled into the exact UTF-8 byte sequences that encode its members
/// (the range-splitting algorithm of Rust's regex-syntax). Shared by the regex and GBNF front ends.
/// </summary>
internal sealed class CharClass
{
    private const int MaxCodePoint = 0x10FFFF;
    private readonly List<(int Lo, int Hi)> _ranges = [];

    /// <summary>Every code point except surrogates.</summary>
    public static CharClass Any() => new CharClass().Add(0, MaxCodePoint);

    /// <summary>Every code point except <c>\n</c> (the regex <c>.</c>).</summary>
    public static CharClass AnyButNewline() => new CharClass().Add('\n').Negate();

    public CharClass Add(int c) => Add(c, c);

    public CharClass Add(int lo, int hi)
    {
        if (lo > hi)
        {
            throw new FormatException($"Invalid character range U+{lo:X4}-U+{hi:X4}.");
        }

        _ranges.Add((lo, hi));
        return this;
    }

    public CharClass Add(CharClass other)
    {
        _ranges.AddRange(other._ranges);
        return this;
    }

    /// <summary>The complement within U+0000..U+10FFFF.</summary>
    public CharClass Negate()
    {
        var merged = Merged();
        var result = new CharClass();
        int next = 0;
        foreach (var (lo, hi) in merged)
        {
            if (lo > next)
            {
                result._ranges.Add((next, lo - 1));
            }

            next = hi + 1;
        }

        if (next <= MaxCodePoint)
        {
            result._ranges.Add((next, MaxCodePoint));
        }

        return result;
    }

    private List<(int Lo, int Hi)> Merged()
    {
        var sorted = _ranges.OrderBy(r => r.Lo).ToList();
        var merged = new List<(int Lo, int Hi)>();
        foreach (var r in sorted)
        {
            if (merged.Count > 0 && r.Lo <= merged[^1].Hi + 1)
            {
                merged[^1] = (merged[^1].Lo, Math.Max(merged[^1].Hi, r.Hi));
            }
            else
            {
                merged.Add(r);
            }
        }

        return merged;
    }

    /// <summary>An expression matching the UTF-8 encoding of exactly one member.</summary>
    /// <exception cref="FormatException">The class is empty.</exception>
    public GrammarExpr ToExpr()
    {
        ByteSet ascii = default;
        var alternatives = new List<GrammarExpr>();
        foreach (var (lo, hi) in Merged())
        {
            // Surrogates have no UTF-8 encoding.
            Split(lo, Math.Min(hi, 0xD7FF), ref ascii, alternatives);
            Split(Math.Max(lo, 0xE000), hi, ref ascii, alternatives);
        }

        if (!ascii.IsEmpty)
        {
            alternatives.Insert(0, GrammarExpr.Bytes(ascii));
        }

        if (alternatives.Count == 0)
        {
            throw new FormatException("The character class matches nothing.");
        }

        return GrammarExpr.Choice(alternatives);
    }

    private static void Split(int lo, int hi, ref ByteSet ascii, List<GrammarExpr> alternatives)
    {
        if (lo > hi)
        {
            return;
        }

        foreach (int max in (ReadOnlySpan<int>)[0x7F, 0x7FF, 0xFFFF])
        {
            if (lo <= max && hi > max)
            {
                Split(lo, max, ref ascii, alternatives);
                Split(max + 1, hi, ref ascii, alternatives);
                return;
            }
        }

        if (hi <= 0x7F)
        {
            ascii = ascii.Union(ByteSet.Range(lo, hi));
            return;
        }

        int length = hi <= 0x7FF ? 2 : hi <= 0xFFFF ? 3 : 4;
        for (int k = 1; k < length; k++)
        {
            int mask = (1 << (6 * k)) - 1;
            if ((lo & ~mask) != (hi & ~mask))
            {
                if ((lo & mask) != 0)
                {
                    Split(lo, lo | mask, ref ascii, alternatives);
                    Split((lo | mask) + 1, hi, ref ascii, alternatives);
                    return;
                }

                if ((hi & mask) != mask)
                {
                    Split(lo, (hi & ~mask) - 1, ref ascii, alternatives);
                    Split(hi & ~mask, hi, ref ascii, alternatives);
                    return;
                }
            }
        }

        byte[] a = Encode(lo);
        byte[] b = Encode(hi);
        var parts = new GrammarExpr[a.Length];
        for (int i = 0; i < a.Length; i++)
        {
            parts[i] = GrammarExpr.Bytes(ByteSet.Range(a[i], b[i]));
        }

        alternatives.Add(GrammarExpr.Seq(parts));
    }

    private static byte[] Encode(int codePoint) => Encoding.UTF8.GetBytes(char.ConvertFromUtf32(codePoint));
}
