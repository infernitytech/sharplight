namespace Sharplight.Tensors;

/// <summary>
/// A tensor dimension: either a fixed extent or a named symbol (e.g. <c>T</c> = number of tokens in
/// the current batch) that is bound when a graph is executed.
/// </summary>
public readonly record struct Dim
{
    /// <summary>The extent of a fixed dimension; 0 for a symbolic one.</summary>
    public long Fixed { get; }
    /// <summary>Name of the symbol for a symbolic dimension (for example <c>T</c>), otherwise <see langword="null"/>.</summary>
    public string? Symbol { get; }
    /// <summary>Upper bound for symbolic dims (used for memory planning). 0 = unknown.</summary>
    public bool IsSymbolic => Symbol is not null;

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
    /// <param name="symbol">Symbol name; its extent is bound per execution (<see cref="Runtime.RunParameters.Symbols"/>) and bounded at compile time (<see cref="Runtime.CompileOptions.Bound"/>).</param>
    public static Dim Sym(string symbol) => new(0, symbol);

    /// <summary>Converts an extent to a fixed dimension.</summary>
    public static implicit operator Dim(long value) => Of(value);
    /// <summary>Converts an extent to a fixed dimension.</summary>
    public static implicit operator Dim(int value) => Of(value);

    /// <summary>
    /// Gets the concrete extent.
    /// </summary>
    /// <param name="bindings">Values of the symbols, or <see langword="null"/> when none are bound.</param>
    /// <exception cref="InvalidOperationException">The dimension is symbolic and not bound.</exception>
    public long Resolve(IReadOnlyDictionary<string, long>? bindings)
    {
        if (Symbol is null)
        {
            return Fixed;
        }

        if (bindings is not null && bindings.TryGetValue(Symbol, out var v))
        {
            return v;
        }

        throw new InvalidOperationException($"Symbolic dimension '{Symbol}' is not bound.");
    }

    /// <inheritdoc/>
    public override string ToString() => Symbol ?? Fixed.ToString();
}
