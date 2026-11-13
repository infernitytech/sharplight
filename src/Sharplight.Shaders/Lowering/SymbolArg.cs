namespace Sharplight.Shaders.Lowering;

/// <summary>Value of a bound symbol (e.g. T) at record time.</summary>
public sealed record SymbolArg(string Symbol, long Multiplier = 1) : KernelArg;
