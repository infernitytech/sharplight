using Sharplight.Tensors;

namespace Sharplight.Onnx;

/// <summary>ONNX TensorProto.DataType ids and helpers.</summary>
internal static class OnnxType
{
    public const int Float = 1, UInt8 = 2, Int8 = 3, UInt16 = 4, Int16 = 5, Int32 = 6, Int64 = 7, String = 8, Bool = 9,
        Float16 = 10, Double = 11, UInt32 = 12, UInt64 = 13, Complex64 = 14, Complex128 = 15, BFloat16 = 16,
        Float8E4M3FN = 17, Float8E4M3FNUZ = 18, Float8E5M2 = 19, Float8E5M2FNUZ = 20, UInt4 = 21, Int4 = 22, Float4E2M1 = 23,
        Float8E8M0 = 24, UInt2 = 25, Int2 = 26;

    /// <summary>Bits per element; 4- and 2-bit types are packed low bits first.</summary>
    public static int ElementBits(int t) => t switch
    {
        UInt4 or Int4 or Float4E2M1 => 4,
        UInt2 or Int2 => 2,
        Float8E4M3FN or Float8E4M3FNUZ or Float8E5M2 or Float8E5M2FNUZ or Float8E8M0 => 8,
        _ => ElementSize(t) * 8,
    };

    /// <summary>Bytes per element of a byte-aligned type (sub-byte types throw; use <see cref="ByteSize"/>).</summary>
    public static int ElementSize(int t) => t switch
    {
        Float or Int32 or UInt32 => 4,
        UInt8 or Int8 or Bool or Float8E4M3FN or Float8E4M3FNUZ or Float8E5M2 or Float8E5M2FNUZ or Float8E8M0 => 1,
        UInt16 or Int16 or Float16 or BFloat16 => 2,
        Int64 or UInt64 or Double or Complex64 => 8,
        Complex128 => 16,
        _ => throw new NotSupportedException($"ONNX element type {Name(t)} has no fixed element size."),
    };

    /// <summary>Bytes of a dense tensor of <paramref name="count"/> elements (sub-byte types packed, the last byte padded).</summary>
    public static long ByteSize(int t, long count) => (count * ElementBits(t) + 7) / 8;

    /// <summary>Whether elements are packed several per byte.</summary>
    public static bool IsSubByte(int t) => t is UInt4 or Int4 or Float4E2M1 or UInt2 or Int2;

    public static bool IsFloat(int t)
        => t is Float or Float16 or Double or BFloat16 or Float8E4M3FN or Float8E4M3FNUZ or Float8E5M2 or Float8E5M2FNUZ or Float4E2M1 or Float8E8M0;

    /// <summary>Signed or unsigned integer types (not bool).</summary>
    public static bool IsInteger(int t) => t is UInt8 or Int8 or UInt16 or Int16 or Int32 or Int64 or UInt32 or UInt64 or UInt4 or Int4 or UInt2 or Int2;

    /// <summary>Unsigned integer types.</summary>
    public static bool IsUnsigned(int t) => t is UInt8 or UInt16 or UInt32 or UInt64 or UInt4 or UInt2;

    /// <summary>
    /// The value set of a type narrower than the IR transport type (f32 for floats, i32 for integers), or null when the transport holds
    /// every value (float, double, int32, int64 and uint64 are carried as is; see docs/onnx-support.md, "Types").
    /// </summary>
    public static NumericFormat? NarrowFormat(int t) => t switch
    {
        Float16 => NumericFormat.F16,
        BFloat16 => NumericFormat.BF16,
        Float8E4M3FN => NumericFormat.F8E4M3FN,
        Float8E4M3FNUZ => NumericFormat.F8E4M3FNUZ,
        Float8E5M2 => NumericFormat.F8E5M2,
        Float8E5M2FNUZ => NumericFormat.F8E5M2FNUZ,
        Float4E2M1 => NumericFormat.F4E2M1,
        Float8E8M0 => NumericFormat.F8E8M0,
        UInt2 => NumericFormat.U2,
        Int2 => NumericFormat.I2,
        UInt4 => NumericFormat.U4,
        Int4 => NumericFormat.I4,
        UInt8 => NumericFormat.U8,
        Int8 => NumericFormat.I8,
        UInt16 => NumericFormat.U16,
        Int16 => NumericFormat.I16,
        _ => null,
    };

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
        Float8E4M3FN => "float8e4m3fn",
        Float8E4M3FNUZ => "float8e4m3fnuz",
        Float8E5M2 => "float8e5m2",
        Float8E5M2FNUZ => "float8e5m2fnuz",
        UInt4 => "uint4",
        Int4 => "int4",
        Float4E2M1 => "float4e2m1",
        Float8E8M0 => "float8e8m0",
        UInt2 => "uint2",
        Int2 => "int2",
        _ => $"type#{t}",
    };
}
