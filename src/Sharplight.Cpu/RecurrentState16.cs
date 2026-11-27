using System.Runtime.CompilerServices;
using Sharplight.Tensors;

namespace Sharplight.Cpu;

/// <summary>
/// 16-bit (bf16 / f16) recurrent state rows for the recurrence ops: a task widens its rows into an f32 copy, updates them in f32,
/// rounds them to the state type (round to nearest even) after every token as the Vulkan kernels do (<c>state16.slang</c>), and
/// narrows them back, which is then exact.
/// </summary>
internal static unsafe class RecurrentState16
{
    /// <summary>Bytes per state element of a recurrent state type: 4 (f32) or 2 (bf16, f16).</summary>
    public static int ElementBytes(DataType type, string op)
        => type == DataType.F32 ? 4 : type == DataType.BF16 || type == DataType.F16 ? 2
            : throw new ArgumentException($"{op}: unsupported state type {type}.");

    /// <summary>Widens <paramref name="n"/> stored elements to f32.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Widen(ushort* src, float* dst, long n, bool bf16)
    {
        if (bf16)
        {
            Simd.BFloatToFloat(src, dst, n);
        }
        else
        {
            Simd.HalfToFloat(src, dst, n);
        }
    }

    /// <summary>Stores <paramref name="n"/> values already rounded to the 16-bit state type (exact).</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Narrow(float* src, ushort* dst, long n, bool bf16)
    {
        for (long i = 0; i < n; i++)
        {
            dst[i] = bf16 ? (ushort)(BitConverter.SingleToUInt32Bits(src[i]) >> 16) : Simd.FloatToHalf(src[i]);
        }
    }

    /// <summary>Rounds <paramref name="n"/> values to the 16-bit state type's precision (round to nearest even), keeping them in f32.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Round(float* row, long n, bool bf16)
    {
        for (long i = 0; i < n; i++)
        {
            row[i] = bf16 ? Simd.BFloatToFloat(Simd.FloatToBFloat(row[i])) : Simd.HalfToFloat(Simd.FloatToHalf(row[i]));
        }
    }
}
