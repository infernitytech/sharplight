using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

/// <summary>Internal data types for repacked weights.</summary>
public static class ShaderTypes
{
    /// <summary>Q8_0 in <see cref="WeightRepack.Q8_0Split"/> layout (same 34 bytes per 32 values).</summary>
    public static readonly DataType Q8_0R = DataType.Register("q8_0r", DataTypeKind.Quantized, 32, 34);
}
