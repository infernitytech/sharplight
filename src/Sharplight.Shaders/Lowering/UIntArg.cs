namespace Sharplight.Shaders.Lowering;

/// <summary>A 32-bit unsigned integer scalar argument.</summary>
/// <param name="Value">The value.</param>
public sealed record UIntArg(uint Value) : KernelArg;
