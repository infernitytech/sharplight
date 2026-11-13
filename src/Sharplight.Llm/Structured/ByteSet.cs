namespace Sharplight.Llm.Structured;

/// <summary>An immutable set of byte values (256 bits).</summary>
internal readonly record struct ByteSet(ulong W0, ulong W1, ulong W2, ulong W3)
{
    /// <summary>All 256 byte values.</summary>
    public static ByteSet All { get; } = new(ulong.MaxValue, ulong.MaxValue, ulong.MaxValue, ulong.MaxValue);

    /// <summary>The single byte <paramref name="b"/>.</summary>
    public static ByteSet Of(byte b) => Range(b, b);

    /// <summary>The bytes <paramref name="lo"/>..<paramref name="hi"/> inclusive (empty when lo &gt; hi).</summary>
    public static ByteSet Range(int lo, int hi)
    {
        ulong w0 = 0, w1 = 0, w2 = 0, w3 = 0;
        for (int b = Math.Max(0, lo); b <= Math.Min(255, hi); b++)
        {
            ulong bit = 1UL << (b & 63);
            switch (b >> 6)
            {
                case 0:
                    w0 |= bit;
                    break;
                case 1:
                    w1 |= bit;
                    break;
                case 2:
                    w2 |= bit;
                    break;
                default:
                    w3 |= bit;
                    break;
            }
        }

        return new ByteSet(w0, w1, w2, w3);
    }

    public bool IsEmpty => (W0 | W1 | W2 | W3) == 0;

    public bool Contains(byte b) => ((b >> 6) switch { 0 => W0, 1 => W1, 2 => W2, _ => W3 } & (1UL << (b & 63))) != 0;

    public ByteSet Union(ByteSet o) => new(W0 | o.W0, W1 | o.W1, W2 | o.W2, W3 | o.W3);

    public ByteSet Except(ByteSet o) => new(W0 & ~o.W0, W1 & ~o.W1, W2 & ~o.W2, W3 & ~o.W3);

    /// <summary>Compact text such as <c>[0-9a-f]</c>, for grammar dumps and cache keys.</summary>
    public override string ToString()
    {
        var sb = new System.Text.StringBuilder("[");
        int b = 0;
        while (b < 256)
        {
            if (!Contains((byte)b))
            {
                b++;
                continue;
            }

            int start = b;
            while (b + 1 < 256 && Contains((byte)(b + 1)))
            {
                b++;
            }

            sb.Append(Show(start));
            if (b > start)
            {
                sb.Append('-').Append(Show(b));
            }

            b++;
        }

        return sb.Append(']').ToString();
    }

    private static string Show(int b) => b is > 0x20 and < 0x7f and not ('-' or ']' or '\\' or '[') ? ((char)b).ToString() : $"\\x{b:X2}";
}
