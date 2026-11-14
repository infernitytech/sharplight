namespace Sharplight.Onnx;

/// <summary>ONNX TensorProto.DataType ids and helpers.</summary>
internal static class OnnxType
{
    public const int Float = 1, UInt8 = 2, Int8 = 3, UInt16 = 4, Int16 = 5, Int32 = 6, Int64 = 7, String = 8, Bool = 9,
        Float16 = 10, Double = 11, UInt32 = 12, UInt64 = 13, Complex64 = 14, Complex128 = 15, BFloat16 = 16;

    public static int ElementSize(int t) => t switch
    {
        Float or Int32 or UInt32 => 4,
        UInt8 or Int8 or Bool => 1,
        UInt16 or Int16 or Float16 or BFloat16 => 2,
        Int64 or UInt64 or Double or Complex64 => 8,
        Complex128 => 16,
        _ => throw new NotSupportedException($"ONNX element type {t} has no fixed element size."),
    };

    public static bool IsFloat(int t) => t is Float or Float16 or Double or BFloat16;

    /// <summary>Signed or unsigned integer types (not bool).</summary>
    public static bool IsInteger(int t) => t is UInt8 or Int8 or UInt16 or Int16 or Int32 or Int64 or UInt32 or UInt64;

    public static string Name(int t) => t switch
    {
        Float => "float",
        UInt8 => "uint8",
        Int8 => "int8",
        UInt16 => "uint16",
        Int16 => "int16",
        Int32 => "int32",
        Int64 => "int64",
        String => "string",
        Bool => "bool",
        Float16 => "float16",
        Double => "double",
        UInt32 => "uint32",
        UInt64 => "uint64",
        BFloat16 => "bfloat16",
        _ => $"type#{t}",
    };
}
