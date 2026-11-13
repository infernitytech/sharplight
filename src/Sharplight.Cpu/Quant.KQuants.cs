using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace Sharplight.Cpu;

// 2-bit and 3-bit k-quants (ggml Q2_K, Q3_K). The arithmetic is not fused so the results match ggml's reference decoders bit for bit.
internal static unsafe partial class Quant
{
    /// <summary>dst[0..16) = v[i] * scale - subtrahend, with separate (unfused) multiply and subtract.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void StoreU8Scaled(Vector128<byte> v, float scale, float subtrahend, float* dst)
    {
        var s = Vector128.Create(scale);
        var m = Vector128.Create(subtrahend);
        var lo = Vector128.WidenLower(v);
        var hi = Vector128.WidenUpper(v);
        (Vector128.ConvertToSingle(Vector128.WidenLower(lo)) * s - m).Store(dst);
        (Vector128.ConvertToSingle(Vector128.WidenUpper(lo)) * s - m).Store(dst + 4);
        (Vector128.ConvertToSingle(Vector128.WidenLower(hi)) * s - m).Store(dst + 8);
        (Vector128.ConvertToSingle(Vector128.WidenUpper(hi)) * s - m).Store(dst + 12);
    }

    /// <summary>dst[0..16) = v[i] * scale for signed bytes.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void StoreI8Scaled(Vector128<sbyte> v, float scale, float* dst)
    {
        var s = Vector128.Create(scale);
        var lo = Vector128.WidenLower(v);
        var hi = Vector128.WidenUpper(v);
        (Vector128.ConvertToSingle(Vector128.WidenLower(lo)) * s).Store(dst);
        (Vector128.ConvertToSingle(Vector128.WidenUpper(lo)) * s).Store(dst + 4);
        (Vector128.ConvertToSingle(Vector128.WidenLower(hi)) * s).Store(dst + 8);
        (Vector128.ConvertToSingle(Vector128.WidenUpper(hi)) * s).Store(dst + 12);
    }

    /// <summary>
    /// Q2_K: 256 values = { u8 scales[16]; u8 qs[64]; f16 d; f16 dmin } (84 bytes). Sixteen 16-value sub-blocks, each with a 4-bit scale
    /// (low nibble) and 4-bit minimum (high nibble); value = d * scale * q2 - dmin * min.
    /// </summary>
    private static void Q2_K(byte* p, long nb, float* dst)
    {
        var three = Vector128.Create((byte)3);
        for (long b = 0; b < nb; b++, p += 84, dst += 256)
        {
            byte* scales = p;
            float d = H(p + 80);
            float min = H(p + 82);
            for (int n = 0; n < 2; n++)
            {
                byte* q = p + 16 + 32 * n;
                float* y = dst + 128 * n;
                for (int j = 0; j < 4; j++)
                {
                    for (int h = 0; h < 2; h++)
                    {
                        int sc = scales[n * 8 + 2 * j + h];
                        float dl = d * (sc & 0xF);
                        float ml = min * (sc >> 4);
                        var v = Vector128.ShiftRightLogical(Vector128.Load(q + 16 * h), 2 * j) & three;
                        StoreU8Scaled(v, dl, ml, y + 32 * j + 16 * h);
                    }
                }
            }
        }
    }

    /// <summary>
    /// Q3_K: 256 values = { u8 hmask[32]; u8 qs[64]; u8 scales[12]; f16 d } (110 bytes). Sixteen 16-value sub-blocks with 6-bit scales
    /// (minus 32); the 3-bit value is the 2-bit code from qs minus 4 where the matching hmask bit is clear.
    /// </summary>
    private static void Q3_K(byte* p, long nb, float* dst)
    {
        var three = Vector128.Create((byte)3);
        var one = Vector128.Create((byte)1);
        var four = Vector128.Create((byte)4);
        uint* aux = stackalloc uint[4];
        sbyte* scales = (sbyte*)aux;
        for (long b = 0; b < nb; b++, p += 110, dst += 256)
        {
            byte* hm = p;
            uint a0 = Unsafe.ReadUnaligned<uint>(p + 96);
            uint a1 = Unsafe.ReadUnaligned<uint>(p + 100);
            uint tmp = Unsafe.ReadUnaligned<uint>(p + 104);
            aux[0] = (a0 & 0x0F0F0F0F) | (((tmp >> 0) & 0x03030303) << 4);
            aux[1] = (a1 & 0x0F0F0F0F) | (((tmp >> 2) & 0x03030303) << 4);
            aux[2] = ((a0 >> 4) & 0x0F0F0F0F) | (((tmp >> 4) & 0x03030303) << 4);
            aux[3] = ((a1 >> 4) & 0x0F0F0F0F) | (((tmp >> 6) & 0x03030303) << 4);
            float dAll = H(p + 108);
            for (int n = 0; n < 2; n++)
            {
                byte* q = p + 32 + 32 * n;
                float* y = dst + 128 * n;
                for (int j = 0; j < 4; j++)
                {
                    for (int h = 0; h < 2; h++)
                    {
                        float dl = dAll * (scales[n * 8 + 2 * j + h] - 32);
                        var low = Vector128.ShiftRightLogical(Vector128.Load(q + 16 * h), 2 * j) & three;
                        var high = Vector128.ShiftRightLogical(Vector128.Load(hm + 16 * h), 4 * n + j) & one;
                        var v = (low + Vector128.ShiftLeft(high, 2) - four).AsSByte();
                        StoreI8Scaled(v, dl, y + 32 * j + 16 * h);
                    }
                }
            }
        }
    }
}
