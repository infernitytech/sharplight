namespace Sharplight.Shaders.Lowering;

/// <summary>A 32-bit float scalar argument.</summary>
/// <param name="Value">The value.</param>
public sealed record FloatArg(float Value) : KernelArg;
