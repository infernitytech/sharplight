namespace Sharplight.Shaders.Lowering;

/// <summary>Binding of symbolic dimensions for one recorded execution.</summary>
public sealed class Bindings(IReadOnlyDictionary<string, long> symbols)
{
    public long this[string symbol] => symbols.TryGetValue(symbol, out var v) ? v : throw new KeyNotFoundException($"Symbol '{symbol}' not bound.");
    public long Get(string? symbol, long fallback) => symbol is null ? fallback : this[symbol];
    public IReadOnlyDictionary<string, long> Values => symbols;
}
