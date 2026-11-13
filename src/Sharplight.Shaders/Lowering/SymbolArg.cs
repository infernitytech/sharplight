namespace Sharplight.Shaders.Lowering;

/// <summary>Value of a bound symbol (e.g. T) at record time.</summary>
/// <param name="Symbol">The symbol name.</param>
/// <param name="Multiplier">Factor applied to the bound value (elements per symbol step).</param>
public sealed record SymbolArg(string Symbol, long Multiplier = 1) : KernelArg;
