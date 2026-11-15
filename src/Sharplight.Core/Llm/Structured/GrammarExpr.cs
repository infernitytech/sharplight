using System.Text;

namespace Sharplight.Llm.Structured;

/// <summary>
/// A byte-level grammar expression: the common target of the GBNF, regex and JSON-schema front ends, compiled by
/// <see cref="GrammarBuilder"/> into a <see cref="Grammar"/>. Immutable.
/// </summary>
internal sealed class GrammarExpr
{
    public enum ExprKind
    {
        Bytes,
        Sequence,
        Choice,
        Reference,
        Repeat,
    }

    private GrammarExpr(ExprKind kind)
    {
        Kind = kind;
    }

    public ExprKind Kind { get; }
    /// <summary>The matched byte for <see cref="ExprKind.Bytes"/>.</summary>
    public ByteSet Set { get; private init; }
    /// <summary>Parts of a sequence or alternatives of a choice; the repeated expression of a repeat.</summary>
    public IReadOnlyList<GrammarExpr> Items { get; private init; } = [];
    /// <summary>Rule name of a reference.</summary>
    public string Name { get; private init; } = "";
    public int Min { get; private init; }
    /// <summary>Upper bound of a repeat; -1 when unbounded.</summary>
    public int Max { get; private init; }

    /// <summary>The empty sequence (matches the empty string).</summary>
    public static GrammarExpr Empty { get; } = new(ExprKind.Sequence);

    public bool IsEmpty => Kind == ExprKind.Sequence && Items.Count == 0;

    public static GrammarExpr Bytes(ByteSet set) => new(ExprKind.Bytes) { Set = set };

    public static GrammarExpr Byte(byte b) => Bytes(ByteSet.Of(b));

    public static GrammarExpr Ref(string name) => new(ExprKind.Reference) { Name = name };

    /// <summary>The UTF-8 bytes of <paramref name="text"/>.</summary>
    public static GrammarExpr Literal(string text) => LiteralBytes(Encoding.UTF8.GetBytes(text));

    public static GrammarExpr LiteralBytes(ReadOnlySpan<byte> bytes)
    {
        var items = new GrammarExpr[bytes.Length];
        for (int i = 0; i < bytes.Length; i++)
        {
            items[i] = Byte(bytes[i]);
        }

        return Seq(items);
    }

    /// <summary>A sequence; nested sequences are flattened and a single part is returned as is.</summary>
    public static GrammarExpr Seq(params IEnumerable<GrammarExpr> items)
    {
        var flat = new List<GrammarExpr>();
        foreach (var item in items)
        {
            if (item.Kind == ExprKind.Sequence)
            {
                flat.AddRange(item.Items);
            }
            else
            {
                flat.Add(item);
            }
        }

        return flat.Count == 1 ? flat[0] : new GrammarExpr(ExprKind.Sequence) { Items = flat };
    }

    /// <summary>Alternatives; nested choices are flattened and a single alternative is returned as is.</summary>
    public static GrammarExpr Choice(params IEnumerable<GrammarExpr> items)
    {
        var flat = new List<GrammarExpr>();
        foreach (var item in items)
        {
            if (item.Kind == ExprKind.Choice)
            {
                flat.AddRange(item.Items);
            }
            else
            {
                flat.Add(item);
            }
        }

        if (flat.Count == 0)
        {
            throw new ArgumentException("A choice needs at least one alternative.");
        }

        return flat.Count == 1 ? flat[0] : new GrammarExpr(ExprKind.Choice) { Items = flat };
    }

    /// <summary><paramref name="item"/> repeated <paramref name="min"/>..<paramref name="max"/> times (max -1: unbounded).</summary>
    public static GrammarExpr Repeat(GrammarExpr item, int min, int max)
    {
        if (min < 0 || (max >= 0 && max < min))
        {
            throw new ArgumentException($"Invalid repetition bounds {{{min},{max}}}.");
        }

        if (max == 0 || item.IsEmpty)
        {
            return Empty;
        }

        if (min == 1 && max == 1)
        {
            return item;
        }

        return new GrammarExpr(ExprKind.Repeat) { Items = [item], Min = min, Max = max };
    }

    public static GrammarExpr Optional(GrammarExpr item) => Repeat(item, 0, 1);

    public static GrammarExpr Star(GrammarExpr item) => Repeat(item, 0, -1);

    public static GrammarExpr Plus(GrammarExpr item) => Repeat(item, 1, -1);
}
