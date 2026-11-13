namespace Sharplight.Shaders.Lowering;

/// <summary>Argument of a kernel launch. Packed in declaration order: pointers (8 bytes) then 32-bit scalars.</summary>
public abstract record KernelArg;
