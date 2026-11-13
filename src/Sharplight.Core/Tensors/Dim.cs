namespace Sharplight.Tensors;

/// <summary>
/// A tensor dimension: either a fixed extent or a named symbol (e.g. <c>T</c> = number of tokens in
/// the current batch) that is bound when a graph is executed.
/// </summary>
public readonly record struct Dim
{
    public long Fixed { get; }
    public string? Symbol { get; }
    /// <summary>Upper bound for symbolic dims (used for memory planning). 0 = unknown.</summary>
    public bool IsSymbolic => Symbol is not null;

    private Dim(long value, string? symbol)
    {
        Fixed = value;
        Symbol = symbol;
    }

    public static Dim Of(long value)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(value);
        return new Dim(value, null);
    }

    public static Dim Sym(string symbol) => new(0, symbol);

    public static implicit operator Dim(long value) => Of(value);
    public static implicit operator Dim(int value) => Of(value);

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

    public override string ToString() => Symbol ?? Fixed.ToString();
}
