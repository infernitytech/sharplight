using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

/// <summary>Internal data types for repacked weights.</summary>
public static class ShaderTypes
{
    /// <summary>Q8_0 in <see cref="WeightRepack.Q8_0Split"/> layout (same 34 bytes per 32 values).</summary>
    public static readonly DataType Q8_0R = DataType.Register("q8_0r", DataTypeKind.Quantized, 32, 34);

    /// <summary>MXFP4 in <see cref="WeightRepack.Mxfp4Split"/> layout (same 17 bytes per 32 values).</summary>
    public static readonly DataType MXFP4R = DataType.Register("mxfp4r", DataTypeKind.Quantized, 32, 17);

    /// <summary>F8E4M3_B32 in <see cref="WeightRepack.Fp8B32Split"/> layout (same 36 bytes per 32 values).</summary>
    public static readonly DataType F8E4M3_B32R = DataType.Register("f8e4m3_b32r", DataTypeKind.Quantized, 32, 36);

    /// <summary>F8E4M3_B128 in <see cref="WeightRepack.Fp8B128Split"/> layout (same 132 bytes per 128 values).</summary>
    public static readonly DataType F8E4M3_B128R = DataType.Register("f8e4m3_b128r", DataTypeKind.Quantized, 128, 132);
}
