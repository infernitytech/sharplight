using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

/// <summary>Binding of symbolic dimensions for one recorded execution.</summary>
public sealed class Bindings(IReadOnlyDictionary<string, long> symbols)
{
    /// <summary>The bound value of <paramref name="symbol"/>; an expression (<c>12*N</c>) is evaluated from the bindings of its symbols.</summary>
    /// <exception cref="KeyNotFoundException">The symbol is not bound.</exception>
    public long this[string symbol] => SymbolicExpression.TryEvaluate(symbol, symbols, out var v) ? v : throw new KeyNotFoundException($"Symbol '{symbol}' not bound.");
    /// <summary>The bound value of <paramref name="symbol"/>, or <paramref name="fallback"/> when <paramref name="symbol"/> is null (a static dimension).</summary>
    public long Get(string? symbol, long fallback) => symbol is null ? fallback : this[symbol];
    /// <summary>All bound symbols and their values.</summary>
    public IReadOnlyDictionary<string, long> Values => symbols;
}
