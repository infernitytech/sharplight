namespace Sharplight.Tensors;

/// <summary>
/// A tensor dimension: either a fixed extent or a symbolic one that is bound when a graph is executed. A symbolic
/// dimension is a named symbol (e.g. <c>T</c> = number of tokens in the current batch) or an expression over symbols
/// (<c>12*N</c>, <c>N+1</c>, <c>ceil(N/32)</c>, see <see cref="SymbolicExpression"/>) whose canonical text is its name.
/// </summary>
public readonly record struct Dim
{
    /// <summary>The extent of a fixed dimension; 0 for a symbolic one.</summary>
    public long Fixed { get; }
    /// <summary>
    /// Name of the symbol (for example <c>T</c>) or canonical text of the expression of a symbolic dimension, otherwise <see langword="null"/>.
    /// </summary>
    public string? Symbol { get; }
    /// <summary>True for a symbolic dimension (a symbol or an expression).</summary>
    public bool IsSymbolic => Symbol is not null;

    /// <summary>The dimension as an expression (a constant for a fixed dimension).</summary>
    public SymbolicExpression Expression => Symbol is null ? SymbolicExpression.Constant(Fixed) : SymbolicExpression.Parse(Symbol);

    private Dim(long value, string? symbol)
    {
        Fixed = value;
        Symbol = symbol;
    }

    /// <summary>
    /// Creates a fixed dimension.
    /// </summary>
    /// <param name="value">The extent.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="value"/> is negative.</exception>
    public static Dim Of(long value)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(value);
        return new Dim(value, null);
    }

    /// <summary>
    /// Creates a symbolic dimension.
    /// </summary>
    /// <param name="symbol">
    /// Symbol name, or the text of an expression over symbols (canonicalized; a constant expression gives a fixed dimension). Symbols are
    /// bound per execution (<see cref="Runtime.RunParameters.Symbols"/>) and bounded at compile time (<see cref="Runtime.CompileOptions.Bound"/>).
    /// </param>
    /// <exception cref="FormatException"><paramref name="symbol"/> is neither a symbol name nor a valid expression.</exception>
    public static Dim Sym(string symbol) => SymbolicExpression.IsSymbolName(symbol) ? new(0, symbol) : FromExpression(SymbolicExpression.Parse(symbol));

    /// <summary>
    /// The dimension of an expression: fixed when it is constant, symbolic otherwise.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">The expression is a negative constant.</exception>
    public static Dim FromExpression(SymbolicExpression expression) => expression.IsConstant ? Of(expression.ConstantValue) : new(0, expression.ToString());

    /// <summary>Product of two dimensions.</summary>
    public static Dim operator *(Dim a, Dim b) => FromExpression(a.Expression * b.Expression);
    /// <summary>Sum of two dimensions.</summary>
    public static Dim operator +(Dim a, Dim b) => FromExpression(a.Expression + b.Expression);
    /// <summary>Difference of two dimensions (must not be negative).</summary>
    public static Dim operator -(Dim a, Dim b) => FromExpression(a.Expression - b.Expression);
    /// <summary><c>floor(this / divisor)</c>.</summary>
    public Dim FloorDiv(long divisor) => FromExpression(Expression.FloorDiv(divisor));
    /// <summary><c>ceil(this / divisor)</c>.</summary>
    public Dim CeilDiv(long divisor) => FromExpression(Expression.CeilDiv(divisor));
    /// <summary>The smaller of two dimensions.</summary>
    public static Dim Min(Dim a, Dim b) => FromExpression(SymbolicExpression.Min(a.Expression, b.Expression));
    /// <summary>The larger of two dimensions.</summary>
    public static Dim Max(Dim a, Dim b) => FromExpression(SymbolicExpression.Max(a.Expression, b.Expression));

    /// <summary>Converts an extent to a fixed dimension.</summary>
    public static implicit operator Dim(long value) => Of(value);
    /// <summary>Converts an extent to a fixed dimension.</summary>
    public static implicit operator Dim(int value) => Of(value);

    /// <summary>
    /// Gets the concrete extent.
    /// </summary>
    /// <param name="bindings">Values of the symbols, or <see langword="null"/> when none are bound.</param>
    /// <exception cref="InvalidOperationException">The dimension is symbolic and a symbol it uses is not bound.</exception>
    public long Resolve(IReadOnlyDictionary<string, long>? bindings)
    {
        if (Symbol is null)
        {
            return Fixed;
        }

        if (bindings is null)
        {
            throw new InvalidOperationException($"Symbolic dimension '{Symbol}' is not bound.");
        }

        return SymbolicExpression.Evaluate(Symbol, bindings);
    }

    /// <inheritdoc/>
    public override string ToString() => Symbol ?? Fixed.ToString();
}
