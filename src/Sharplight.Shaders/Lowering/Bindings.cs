using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

/// <summary>
/// Binding of symbolic dimensions for one recorded execution. The symbol values must not change while it is in use: evaluated
/// expressions are memoized (recording evaluates the same count expressions for every launch argument and grid).
/// </summary>
public sealed class Bindings(IReadOnlyDictionary<string, long> symbols)
{
    private readonly System.Collections.Concurrent.ConcurrentDictionary<string, long> _evaluated = new(StringComparer.Ordinal);
    private readonly System.Collections.Concurrent.ConcurrentDictionary<Func<Bindings, bool>, bool> _conditions = new(ReferenceEqualityComparer.Instance);

    /// <summary>The bound value of <paramref name="symbol"/>; an expression (<c>12*N</c>) is evaluated from the bindings of its symbols.</summary>
    /// <exception cref="KeyNotFoundException">The symbol is not bound.</exception>
    public long this[string symbol]
    {
        get
        {
            if (_evaluated.TryGetValue(symbol, out long cached))
            {
                return cached;
            }

            long v = SymbolicExpression.TryEvaluate(symbol, symbols, out var value) ? value : throw new KeyNotFoundException($"Symbol '{symbol}' not bound.");
            _evaluated[symbol] = v;
            return v;
        }
    }
    /// <summary>
    /// The value of <paramref name="condition"/> for these bindings, evaluated once per condition (by reference) and remembered: a condition
    /// shared by many variants (a launch condition, the guard of a fused launch's fallback variants) is asked once per launch and variant
    /// when a step shape is recorded, and some evaluate the selection of other launches themselves. Conditions must depend on the
    /// bindings only.
    /// </summary>
    public bool Holds(Func<Bindings, bool> condition) => _conditions.GetOrAdd(condition, static (c, b) => c(b), this);

    /// <summary>The bound value of <paramref name="symbol"/>, or <paramref name="fallback"/> when <paramref name="symbol"/> is null (a static dimension).</summary>
    public long Get(string? symbol, long fallback) => symbol is null ? fallback : this[symbol];
    /// <summary>All bound symbols and their values.</summary>
    public IReadOnlyDictionary<string, long> Values => symbols;
}
