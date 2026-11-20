using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Sharplight.Tensors.Grids;

namespace Sharplight.Cpu;

// Fused integer dots of ggml's grid formats (and MXFP4, at the end) with an 8-bit activation row (AVX2), llama.cpp's approach: per 32 values the grid magnitudes
// (unsigned, below 64) are multiplied with the activations whose signs the sign pattern flipped (vpsignb; vpmaddubsw cannot saturate:
// 2 * 62 * 127 < 32768), summed to int32 lanes and scaled once in float. No decoded row is written, unlike Decode + Dot.
internal static unsafe partial class IntDot
{
    /// <summary>Whether <paramref name="f"/> has a fused dot (<see cref="FusedDot"/>) on this host: the grid formats with AVX2.</summary>
    public static bool HasFusedDot(WeightFormat f) => CpuIsa.Use256 && Avx2.IsSupported
        && f is WeightFormat.IQ2_XXS or WeightFormat.IQ2_XS or WeightFormat.IQ2_S or WeightFormat.IQ3_XXS or WeightFormat.IQ3_S or WeightFormat.MXFP4;

    /// <summary>Dot product of one weight row in format <paramref name="f"/> (<see cref="HasFusedDot"/>) with a quantized activation row.</summary>
    /// <param name="f">The weight format.</param>
    /// <param name="row">The weight row, <paramref name="k"/> values.</param>
    /// <param name="x">The activation row.</param>
    /// <param name="k">Row length, a multiple of the format's block (256; 32 for MXFP4).</param>
    public static float FusedDot(WeightFormat f, byte* row, Q8Row x, int k) => f switch
    {
        WeightFormat.IQ2_XXS => DotIq2XxsAvx2(row, x, k),
        WeightFormat.IQ2_XS => DotIq2XsAvx2(row, x, k),
        WeightFormat.IQ2_S => DotIq2SAvx2(row, x, k),
        WeightFormat.IQ3_XXS => DotIq3XxsAvx2(row, x, k),
        WeightFormat.IQ3_S => DotIq3SAvx2(row, x, k),
        WeightFormat.MXFP4 => DotMxfp4Avx2(row, x, k),
        _ => throw new NotSupportedException($"{f} has no fused integer dot."),
    };

    /// <summary>[256]: per sign pattern eight bytes of +1, or -1 where its bit is set (the sign operand of <c>vpsignb</c>).</summary>
    private static readonly ulong[] s_iqSignBytes = BuildSignBytes();

    /// <summary>[128]: <see cref="s_iqSignBytes"/> of the eight-bit sign pattern that a seven-bit sign index selects (ggml's ksigns), one lookup.</summary>
    private static readonly ulong[] s_iqSignBytes7 = BuildSignBytes7();

    private static ulong SignBytes(int pattern)
    {
        ulong bytes = 0;
        for (int j = 0; j < 8; j++)
        {
            bytes |= (((pattern >> j) & 1) != 0 ? 0xFFUL : 0x01UL) << (8 * j);
        }

        return bytes;
    }

    private static ulong[] BuildSignBytes()
    {
        var table = new ulong[256];
        for (int i = 0; i < 256; i++)
        {
            table[i] = SignBytes(i);
        }

        return table;
    }

    private static ulong[] BuildSignBytes7()
    {
        var table = new ulong[128];
        var ksigns = IqSigns.Values;
        for (int i = 0; i < 128; i++)
        {
            table[i] = SignBytes(ksigns[i]);
        }

        return table;
    }

    /// <summary>The int32 sums per four values of 32 grid magnitudes times 32 activations with the signs applied (+1 / -1 bytes).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> SignedSums(Vector256<byte> magnitudes, Vector256<sbyte> signs, sbyte* q)
    {
        var signed = Avx2.Sign(Vector256.Load(q), signs);
        return Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(magnitudes, signed), Vector256.Create((short)1));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> Accumulate(Vector256<int> sums, Vector256<float> scale, Vector256<float> acc) =>
        Vector256.FusedMultiplyAdd(Vector256.ConvertToSingle(sums), scale, acc);

    /// <summary>Four 64-bit table entries (grid magnitudes or sign bytes) as one vector, built from two 128-bit halves.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<byte> Entries(ulong* table, uint i0, uint i1, uint i2, uint i3) =>
        Vector256.Create(Vector128.Create(table[i0], table[i1]), Vector128.Create(table[i2], table[i3])).AsByte();

    /// <summary>IQ2_XXS, two sub-blocks of 32 per iteration with their own accumulators (one 64-bit read of indices and signs each).</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static float DotIq2XxsAvx2(byte* p, Q8Row x, int k)
    {
        fixed (ulong* grid = Iq2XxsGrid.Values)
        fixed (ulong* signs7 = s_iqSignBytes7)
        {
            var acc0 = Vector256<float>.Zero;
            var acc1 = Vector256<float>.Zero;
            sbyte* q = x.Q;
            float* scales = x.LaneScales;
            for (int blk = 0; blk < k / 256; blk++, p += 66)
            {
                float d = Quant.H(p) * 0.25f;
                for (int ib32 = 0; ib32 < 8; ib32 += 2, q += 64, scales += 16)
                {
                    ulong w0 = Unsafe.ReadUnaligned<ulong>(p + 2 + 8 * ib32);
                    ulong w1 = Unsafe.ReadUnaligned<ulong>(p + 10 + 8 * ib32);
                    uint i0 = (uint)w0;
                    uint s0 = (uint)(w0 >> 32);
                    uint i1 = (uint)w1;
                    uint s1 = (uint)(w1 >> 32);
                    var m0 = Entries(grid, i0 & 0xFF, (i0 >> 8) & 0xFF, (i0 >> 16) & 0xFF, i0 >> 24);
                    var g0 = Entries(signs7, s0 & 127, (s0 >> 7) & 127, (s0 >> 14) & 127, (s0 >> 21) & 127).AsSByte();
                    var m1 = Entries(grid, i1 & 0xFF, (i1 >> 8) & 0xFF, (i1 >> 16) & 0xFF, i1 >> 24);
                    var g1 = Entries(signs7, s1 & 127, (s1 >> 7) & 127, (s1 >> 14) & 127, (s1 >> 21) & 127).AsSByte();
                    acc0 = Accumulate(SignedSums(m0, g0, q), Vector256.Create(d * (0.5f + (s0 >> 28)) * scales[0]), acc0);
                    acc1 = Accumulate(SignedSums(m1, g1, q + 32), Vector256.Create(d * (0.5f + (s1 >> 28)) * scales[8]), acc1);
                }
            }

            return Vector256.Sum(acc0 + acc1);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static float DotIq2XsAvx2(byte* p, Q8Row x, int k)
    {
        fixed (ulong* grid = Iq2XsGrid.Values)
        fixed (ulong* signs7 = s_iqSignBytes7)
        {
            var acc = Vector256<float>.Zero;
            sbyte* q = x.Q;
            float* scales = x.LaneScales;
            for (int blk = 0; blk < k / 256; blk++, p += 74)
            {
                float d = Quant.H(p) * 0.25f;
                ushort* qs = (ushort*)(p + 2);
                for (int ib32 = 0; ib32 < 8; ib32++, q += 32, scales += 8, qs += 4)
                {
                    int q0 = Unsafe.ReadUnaligned<ushort>(qs);
                    int q1 = Unsafe.ReadUnaligned<ushort>(qs + 1);
                    int q2 = Unsafe.ReadUnaligned<ushort>(qs + 2);
                    int q3 = Unsafe.ReadUnaligned<ushort>(qs + 3);
                    var magnitudes = Vector256.Create(grid[q0 & 511], grid[q1 & 511], grid[q2 & 511], grid[q3 & 511]).AsByte();
                    var negate = Vector256.Create(signs7[q0 >> 9], signs7[q1 >> 9], signs7[q2 >> 9], signs7[q3 >> 9]).AsSByte();
                    int s = p[66 + ib32];
                    float dx = d * scales[0];
                    var scale = Vector256.Create(Vector128.Create(dx * (0.5f + (s & 0xF))), Vector128.Create(dx * (0.5f + (s >> 4))));
                    acc = Accumulate(SignedSums(magnitudes, negate, q), scale, acc);
                }
            }

            return Vector256.Sum(acc);
        }
    }

    /// <summary>IQ2_S, two sub-blocks of 32 per iteration with their own accumulators.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static float DotIq2SAvx2(byte* p, Q8Row x, int k)
    {
        fixed (ulong* grid = Iq2SGrid.Values)
        fixed (ulong* masks = s_iqSignBytes)
        {
            var acc0 = Vector256<float>.Zero;
            var acc1 = Vector256<float>.Zero;
            sbyte* q = x.Q;
            float* scales = x.LaneScales;
            for (int blk = 0; blk < k / 256; blk++, p += 82)
            {
                float d = Quant.H(p) * 0.25f;
                byte* qs = p + 2;
                byte* signs = qs + 32;
                for (int ib32 = 0; ib32 < 8; ib32 += 2, q += 64, scales += 16, qs += 8, signs += 8)
                {
                    uint h0 = p[66 + ib32];
                    uint h1 = p[67 + ib32];
                    var m0 = Entries(grid, qs[0] | ((h0 << 8) & 0x300), qs[1] | ((h0 << 6) & 0x300), qs[2] | ((h0 << 4) & 0x300), qs[3] | ((h0 << 2) & 0x300));
                    var m1 = Entries(grid, qs[4] | ((h1 << 8) & 0x300), qs[5] | ((h1 << 6) & 0x300), qs[6] | ((h1 << 4) & 0x300), qs[7] | ((h1 << 2) & 0x300));
                    var g0 = Entries(masks, signs[0], signs[1], signs[2], signs[3]).AsSByte();
                    var g1 = Entries(masks, signs[4], signs[5], signs[6], signs[7]).AsSByte();
                    int sc0 = p[74 + ib32];
                    int sc1 = p[75 + ib32];
                    float dx0 = d * scales[0];
                    float dx1 = d * scales[8];
                    acc0 = Accumulate(SignedSums(m0, g0, q), Vector256.Create(Vector128.Create(dx0 * (0.5f + (sc0 & 0xF))), Vector128.Create(dx0 * (0.5f + (sc0 >> 4)))), acc0);
                    acc1 = Accumulate(SignedSums(m1, g1, q + 32), Vector256.Create(Vector128.Create(dx1 * (0.5f + (sc1 & 0xF))), Vector128.Create(dx1 * (0.5f + (sc1 >> 4)))), acc1);
                }
            }

            return Vector256.Sum(acc0 + acc1);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static float DotIq3XxsAvx2(byte* p, Q8Row x, int k)
    {
        fixed (uint* grid = Iq3XxsGrid.Values)
        fixed (ulong* signs7 = s_iqSignBytes7)
        {
            var acc = Vector256<float>.Zero;
            sbyte* q = x.Q;
            float* scales = x.LaneScales;
            for (int blk = 0; blk < k / 256; blk++, p += 98)
            {
                float d = Quant.H(p) * 0.5f;
                byte* qs = p + 2;
                for (int ib32 = 0; ib32 < 8; ib32++, q += 32, scales += 8, qs += 8)
                {
                    uint word = Unsafe.ReadUnaligned<uint>(p + 66 + 4 * ib32);
                    var magnitudes = Vector256.Create(grid[qs[0]], grid[qs[1]], grid[qs[2]], grid[qs[3]], grid[qs[4]], grid[qs[5]], grid[qs[6]], grid[qs[7]]).AsByte();
                    var negate = Vector256.Create(signs7[word & 127], signs7[(word >> 7) & 127], signs7[(word >> 14) & 127], signs7[(word >> 21) & 127]).AsSByte();
                    float scale = d * (0.5f + (word >> 28)) * scales[0];
                    acc = Accumulate(SignedSums(magnitudes, negate, q), Vector256.Create(scale), acc);
                }
            }

            return Vector256.Sum(acc);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static float DotIq3SAvx2(byte* p, Q8Row x, int k)
    {
        fixed (uint* grid = Iq3SGrid.Values)
        fixed (ulong* masks = s_iqSignBytes)
        {
            var acc = Vector256<float>.Zero;
            sbyte* q = x.Q;
            float* scales = x.LaneScales;
            for (int blk = 0; blk < k / 256; blk++, p += 110)
            {
                float d = Quant.H(p);
                byte* qs = p + 2;
                byte* signs = p + 74;
                for (int ib32 = 0; ib32 < 8; ib32++, q += 32, scales += 8, qs += 8, signs += 4)
                {
                    int high = p[66 + ib32];
                    var magnitudes = Vector256.Create(
                        grid[qs[0] | ((high << 8) & 256)], grid[qs[1] | ((high << 7) & 256)],
                        grid[qs[2] | ((high << 6) & 256)], grid[qs[3] | ((high << 5) & 256)],
                        grid[qs[4] | ((high << 4) & 256)], grid[qs[5] | ((high << 3) & 256)],
                        grid[qs[6] | ((high << 2) & 256)], grid[qs[7] | ((high << 1) & 256)]).AsByte();
                    var negate = Vector256.Create(masks[signs[0]], masks[signs[1]], masks[signs[2]], masks[signs[3]]).AsSByte();
                    int s = (p[106 + ib32 / 2] >> (4 * (ib32 & 1))) & 0xF;
                    float scale = d * (1 + 2 * s) * scales[0];
                    acc = Accumulate(SignedSums(magnitudes, negate, q), Vector256.Create(scale), acc);
                }
            }

            return Vector256.Sum(acc);
        }
    }

    /// <summary>The doubled E2M1 values of the 16 MXFP4 codes, signed (scaled by <see cref="Quant.E8M0Half"/>).</summary>
    private static readonly Vector128<sbyte> s_fp4Values = Vector128.Create((sbyte)0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);

    /// <summary>
    /// MXFP4 { u8 e; u8 qs[16] } per 32 values (low nibbles the first 16, high nibbles the next): the codes looked up as signed values
    /// (<c>vpshufb</c>), multiplied as |w| times the activations with w's sign (<c>vpmaddubsw</c>, at most 2 * 12 * 127), one scale per block.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static float DotMxfp4Avx2(byte* p, Q8Row x, int k)
    {
        var table = Vector256.Create(s_fp4Values, s_fp4Values);
        var low = Vector128.Create((byte)0x0F);
        var ones = Vector256.Create((short)1);
        var acc = Vector256<float>.Zero;
        sbyte* q = x.Q;
        float* scales = x.LaneScales;
        for (int blk = 0; blk < k / 32; blk++, p += 17, q += 32, scales += 8)
        {
            var bits = Vector128.Load(p + 1);
            var codes = Vector256.Create(bits & low, Vector128.ShiftRightLogical(bits, 4)).AsSByte();
            var w = Avx2.Shuffle(table, codes);
            var products = Avx2.MultiplyAddAdjacent(Avx2.Abs(w), Avx2.Sign(Vector256.Load(q), w));
            var sums = Avx2.MultiplyAddAdjacent(products, ones);
            acc = Accumulate(sums, Vector256.Create(Quant.E8M0Half(p[0]) * scales[0]), acc);
        }

        return Vector256.Sum(acc);
    }
}
