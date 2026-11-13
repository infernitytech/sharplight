namespace Sharplight.Tensors;

/// <summary>
/// One factor of a monomial in a <see cref="SymbolicExpression"/>: an atomic symbol or an opaque sub-term
/// (<c>floor(e/c)</c>, <c>ceil(e/c)</c>, <c>min(a,b)</c>, <c>max(a,b)</c>). Factors are compared by their canonical text.
/// </summary>
internal sealed class SymbolicFactor
{
    private SymbolicFactor(SymbolicFactorKind kind, string? symbol, SymbolicExpression[] operands, long divisor)
    {
        Kind = kind;
        Symbol = symbol;
        Operands = operands;
        Divisor = divisor;
        Text = kind switch
        {
            SymbolicFactorKind.Symbol => symbol!,
            SymbolicFactorKind.FloorDiv => $"floor({operands[0].ToOperandString()}/{divisor})",
            SymbolicFactorKind.CeilDiv => $"ceil({operands[0].ToOperandString()}/{divisor})",
            SymbolicFactorKind.Min => $"min({operands[0]},{operands[1]})",
            _ => $"max({operands[0]},{operands[1]})",
        };
    }

    /// <summary>What the factor computes.</summary>
    public SymbolicFactorKind Kind { get; }

    /// <summary>Name of an atomic symbol; null for the other kinds.</summary>
    public string? Symbol { get; }

    /// <summary>Operands of a division (one) or of min/max (two, ordered by text).</summary>
    public SymbolicExpression[] Operands { get; }

    /// <summary>Positive divisor of a floor/ceil division.</summary>
    public long Divisor { get; }

    /// <summary>Canonical text; the identity of the factor.</summary>
    public string Text { get; }

    public static SymbolicFactor Atom(string symbol) => new(SymbolicFactorKind.Symbol, symbol, [], 0);

    public static SymbolicFactor Division(SymbolicExpression operand, long divisor, bool ceiling)
        => new(ceiling ? SymbolicFactorKind.CeilDiv : SymbolicFactorKind.FloorDiv, null, [operand], divisor);

    public static SymbolicFactor Extremum(SymbolicExpression a, SymbolicExpression b, bool maximum)
    {
        var (first, second) = string.CompareOrdinal(a.ToString(), b.ToString()) <= 0 ? (a, b) : (b, a);
        return new(maximum ? SymbolicFactorKind.Max : SymbolicFactorKind.Min, null, [first, second], 0);
    }

    /// <summary>Evaluates the factor; <paramref name="atom"/> supplies atomic symbol values.</summary>
    public long Evaluate(Func<string, long> atom) => Kind switch
    {
        SymbolicFactorKind.Symbol => atom(Symbol!),
        SymbolicFactorKind.FloorDiv => Math.DivRem(Operands[0].Evaluate(atom), Divisor) is var (q, r) && r < 0 ? q - 1 : q,
        SymbolicFactorKind.CeilDiv => Math.DivRem(Operands[0].Evaluate(atom), Divisor) is var (q, r) && r > 0 ? q + 1 : q,
        SymbolicFactorKind.Min => Math.Min(Operands[0].Evaluate(atom), Operands[1].Evaluate(atom)),
        _ => Math.Max(Operands[0].Evaluate(atom), Operands[1].Evaluate(atom)),
    };

    /// <summary>Upper bound of the factor given upper bounds of the atoms (every factor kind is monotone).</summary>
    public long UpperBound(Func<string, long> atomBound) => Kind switch
    {
        SymbolicFactorKind.Symbol => atomBound(Symbol!),
        SymbolicFactorKind.FloorDiv or SymbolicFactorKind.CeilDiv => Division(Constant(Operands[0].UpperBound(atomBound)), Divisor, Kind == SymbolicFactorKind.CeilDiv).Evaluate(_ => 0),
        SymbolicFactorKind.Min => Math.Min(Operands[0].UpperBound(atomBound), Operands[1].UpperBound(atomBound)),
        _ => Math.Max(Operands[0].UpperBound(atomBound), Operands[1].UpperBound(atomBound)),
    };

    /// <summary>The atomic symbols the factor depends on.</summary>
    public IEnumerable<string> Atoms() => Kind == SymbolicFactorKind.Symbol ? [Symbol!] : Operands.SelectMany(o => o.Atoms());

    private static SymbolicExpression Constant(long value) => SymbolicExpression.Constant(value);

    /// <inheritdoc />
    public override string ToString() => Text;
}
