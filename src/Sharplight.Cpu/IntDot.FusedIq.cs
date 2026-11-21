using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Sharplight.Tensors.Grids;

namespace Sharplight.Cpu;

// Fused integer dots of ggml's grid formats (and MXFP4, Q4_K and Q6_K, at the end) with an 8-bit activation row (AVX2), llama.cpp's approach: per 32 values the grid magnitudes
// (unsigned, below 64) are multiplied with the activations whose signs the sign pattern flipped (vpsignb; vpmaddubsw cannot saturate:
// 2 * 62 * 127 < 32768), summed to int32 lanes and scaled once in float. No decoded row is written, unlike Decode + Dot.
internal static unsafe partial class IntDot
{
    /// <summary>Whether <paramref name="f"/> has a fused dot (<see cref="FusedDot"/>) on this host: the grid formats with AVX2.</summary>
    public static bool HasFusedDot(WeightFormat f) => CpuIsa.Use256 && Avx2.IsSupported
        && f is WeightFormat.IQ2_XXS or WeightFormat.IQ2_XS or WeightFormat.IQ2_S or WeightFormat.IQ3_XXS or WeightFormat.IQ3_S or WeightFormat.MXFP4
            or WeightFormat.Q4_K or WeightFormat.Q6_K;

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
        WeightFormat.Q4_K => DotQ4KAvx2(row, x, k),
        WeightFormat.Q6_K => DotQ6KAvx2(row, x, k),
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
    /// <summary>
    /// Q4_K { f16 d; f16 dmin; u8 scales[12]; u8 qs[128] } per 256 values, llama.cpp's AVX2 approach: the 4-bit codes (unsigned) times the
    /// activations (<c>vpmaddubsw</c>, at most 2 * 15 * 127) are summed in pairs weighted by the sub-block's 6-bit scale (<c>vpmaddwd</c>),
    /// scaled by the activation lanes in float and by d once per 256 values; the mins term dmin * m * sum(x) comes from the activation
    /// row's group sums, eight sub-blocks per vector. Two sub-blocks per 32 code bytes (low and high nibbles), each with its own accumulator.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static float DotQ4KAvx2(byte* p, Q8Row x, int k)
    {
        var low = Vector256.Create((byte)0x0F);
        var pick0 = Vector256.Create((ushort)0x0100).AsByte();
        var pick1 = Vector256.Create((ushort)0x0302).AsByte();
        var pick2 = Vector256.Create((ushort)0x0504).AsByte();
        var pick3 = Vector256.Create((ushort)0x0706).AsByte();
        var pick4 = Vector256.Create((ushort)0x0908).AsByte();
        var pick5 = Vector256.Create((ushort)0x0B0A).AsByte();
        var pick6 = Vector256.Create((ushort)0x0D0C).AsByte();
        var pick7 = Vector256.Create((ushort)0x0F0E).AsByte();
        var total = Vector256<float>.Zero;
        var mins = Vector256<float>.Zero;
        sbyte* q = x.Q;
        float* laneScales = x.LaneScales;
        float* groupSums = x.GroupSums;
        for (int blk = 0; blk < k / 256; blk++, p += 144, q += 256, laneScales += 64, groupSums += 16)
        {
            float d = Quant.H(p);
            float dmin = Quant.H(p + 2);
            KScales(p + 4, out ulong scaleBytes, out ulong minBytes);
            var scales128 = Sse41.ConvertToVector128Int16(Vector128.CreateScalar(scaleBytes).AsByte());
            var scales = Vector256.Create(scales128, scales128).AsByte();
            var minValues = Vector256.ConvertToSingle(Avx2.ConvertToVector256Int32(Vector128.CreateScalar(minBytes).AsByte()));
            // Sums of the activations per 32-value sub-block, in sub-block order.
            var pairs = Avx.HorizontalAdd(Vector256.Load(groupSums), Vector256.Load(groupSums + 8));
            var subBlockSums = Avx2.Permute4x64(pairs.AsDouble(), 0b11_01_10_00).AsSingle();
            mins = Vector256.FusedMultiplyAdd(minValues * dmin, subBlockSums, mins);
            byte* qs = p + 16;
            var acc0 = Vector256<float>.Zero;
            var acc1 = Vector256<float>.Zero;
            acc0 = Q4KPair(Vector256.Load(qs), q, laneScales, Avx2.Shuffle(scales, pick0), Avx2.Shuffle(scales, pick1), low, acc0, ref acc1);
            acc0 = Q4KPair(Vector256.Load(qs + 32), q + 64, laneScales + 16, Avx2.Shuffle(scales, pick2), Avx2.Shuffle(scales, pick3), low, acc0, ref acc1);
            acc0 = Q4KPair(Vector256.Load(qs + 64), q + 128, laneScales + 32, Avx2.Shuffle(scales, pick4), Avx2.Shuffle(scales, pick5), low, acc0, ref acc1);
            acc0 = Q4KPair(Vector256.Load(qs + 96), q + 192, laneScales + 48, Avx2.Shuffle(scales, pick6), Avx2.Shuffle(scales, pick7), low, acc0, ref acc1);
            total = Vector256.FusedMultiplyAdd(acc0 + acc1, Vector256.Create(d), total);
        }

        return Vector256.Sum(total - mins);
    }

    /// <summary>
    /// Two 32-value sub-blocks of a Q4_K block from one 32-byte load of codes (low nibbles, then high nibbles): the code-activation pair sums
    /// weighted by the sub-blocks' scales (int16 lanes, broadcast), converted and scaled by the activation lanes.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> Q4KPair(Vector256<byte> bits, sbyte* q, float* laneScales, Vector256<byte> scale0, Vector256<byte> scale1, Vector256<byte> low,
        Vector256<float> acc0, ref Vector256<float> acc1)
    {
        var codes0 = bits & low;
        var codes1 = Vector256.ShiftRightLogical(bits.AsUInt16(), 4).AsByte() & low;
        var sums0 = Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(codes0, Vector256.Load(q)), scale0.AsInt16());
        var sums1 = Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(codes1, Vector256.Load(q + 32)), scale1.AsInt16());
        acc1 = Accumulate(sums1, Vector256.Load(laneScales + 8), acc1);
        return Accumulate(sums0, Vector256.Load(laneScales), acc0);
    }

    /// <summary>
    /// The eight 6-bit scales and the eight 6-bit mins of a Q4_K / Q5_K block's 12 packed bytes, one byte each (llama.cpp's unpacking):
    /// sub-blocks 0-3 keep the low six bits of bytes 0-7, sub-blocks 4-7 take four bits from bytes 8-11 and the top two of bytes 0-7.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void KScales(byte* packed, out ulong scales, out ulong mins)
    {
        const uint Low6 = 0x3F3F3F3F;
        const uint Low4 = 0x0F0F0F0F;
        const uint Low2 = 0x03030303;
        uint u0 = Unsafe.ReadUnaligned<uint>(packed);
        uint u1 = Unsafe.ReadUnaligned<uint>(packed + 4);
        uint u2 = Unsafe.ReadUnaligned<uint>(packed + 8);
        uint scalesHigh = (u2 & Low4) | (((u0 >> 6) & Low2) << 4);
        uint minsHigh = ((u2 >> 4) & Low4) | (((u1 >> 6) & Low2) << 4);
        scales = (u0 & Low6) | ((ulong)scalesHigh << 32);
        mins = (u1 & Low6) | ((ulong)minsHigh << 32);
    }

    /// <summary>
    /// Q6_K { u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d } per 256 values, llama.cpp's AVX2 approach: the 6-bit codes (unsigned, below 64)
    /// times the activations (<c>vpmaddubsw</c>, at most 2 * 63 * 127) are summed in pairs weighted by the 16-value groups' scales
    /// (<c>vpmaddwd</c>), scaled by the activation lanes in float and by d once per 256 values; the offset of 32 comes from the activation
    /// row's group sums, eight groups per vector.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static float DotQ6KAvx2(byte* p, Q8Row x, int k)
    {
        var low = Vector256.Create((byte)0x0F);
        var two = Vector256.Create((byte)3);
        // Per 32 values the scales of their two 16-value groups: group 2i in the low 128-bit lane, 2i + 1 in the high one.
        var pick0 = Vector256.Create(Vector128.Create((ushort)0x0100), Vector128.Create((ushort)0x0302)).AsByte();
        var pick1 = Vector256.Create(Vector128.Create((ushort)0x0504), Vector128.Create((ushort)0x0706)).AsByte();
        var pick2 = Vector256.Create(Vector128.Create((ushort)0x0908), Vector128.Create((ushort)0x0B0A)).AsByte();
        var pick3 = Vector256.Create(Vector128.Create((ushort)0x0D0C), Vector128.Create((ushort)0x0F0E)).AsByte();
        var total = Vector256<float>.Zero;
        var offset = Vector256<float>.Zero;
        sbyte* q = x.Q;
        float* laneScales = x.LaneScales;
        float* groupSums = x.GroupSums;
        for (int blk = 0; blk < k / 256; blk++, p += 210, q += 256, laneScales += 64, groupSums += 16)
        {
            float d = Quant.H(p + 208);
            var scaleBytes = Vector128.Load(p + 192);
            var groupScales = Vector256.FusedMultiplyAdd(
                Vector256.ConvertToSingle(Avx2.ConvertToVector256Int32(scaleBytes.AsSByte())), Vector256.Load(groupSums),
                Vector256.ConvertToSingle(Avx2.ConvertToVector256Int32(Sse2.ShiftRightLogical128BitLane(scaleBytes, 8).AsSByte())) * Vector256.Load(groupSums + 8));
            offset = Vector256.FusedMultiplyAdd(groupScales, Vector256.Create(32 * d), offset);
            var scales16 = Avx2.ConvertToVector256Int16(scaleBytes.AsSByte()).AsByte();
            var acc0 = Vector256<float>.Zero;
            var acc1 = Vector256<float>.Zero;
            for (int n = 0; n < 2; n++)
            {
                // The eight scales of these 128 values in both 128-bit lanes.
                var scales = n == 0 ? Avx2.Permute2x128(scales16, scales16, 0x00) : Avx2.Permute2x128(scales16, scales16, 0x11);
                var ql0 = Vector256.Load(p + 64 * n);
                var ql1 = Vector256.Load(p + 64 * n + 32);
                var qh = Vector256.Load(p + 128 + 32 * n);
                var high0 = Vector256.ShiftLeft((qh & two).AsUInt16(), 4).AsByte();
                var high1 = Vector256.ShiftLeft((Vector256.ShiftRightLogical(qh.AsUInt16(), 2).AsByte() & two).AsUInt16(), 4).AsByte();
                var high2 = Vector256.ShiftLeft((Vector256.ShiftRightLogical(qh.AsUInt16(), 4).AsByte() & two).AsUInt16(), 4).AsByte();
                var high3 = Vector256.ShiftLeft((Vector256.ShiftRightLogical(qh.AsUInt16(), 6).AsByte() & two).AsUInt16(), 4).AsByte();
                int base0 = 128 * n;
                acc0 = Q6Part((ql0 & low) | high0, q + base0, laneScales + base0 / 4, Avx2.Shuffle(scales, pick0), acc0);
                acc1 = Q6Part((ql1 & low) | high1, q + base0 + 32, laneScales + base0 / 4 + 8, Avx2.Shuffle(scales, pick1), acc1);
                acc0 = Q6Part((Vector256.ShiftRightLogical(ql0.AsUInt16(), 4).AsByte() & low) | high2, q + base0 + 64, laneScales + base0 / 4 + 16, Avx2.Shuffle(scales, pick2), acc0);
                acc1 = Q6Part((Vector256.ShiftRightLogical(ql1.AsUInt16(), 4).AsByte() & low) | high3, q + base0 + 96, laneScales + base0 / 4 + 24, Avx2.Shuffle(scales, pick3), acc1);
            }

            total = Vector256.FusedMultiplyAdd(acc0 + acc1, Vector256.Create(d), total);
        }

        return Vector256.Sum(total - offset);
    }

    /// <summary>
    /// One 32-value part of a Q6_K dot: the code-activation pair sums weighted by the scales of the two 16-value groups (int16 lanes: the first
    /// group's in the low 128-bit lane, the second's in the high one), converted and scaled by the activation lanes.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> Q6Part(Vector256<byte> codes, sbyte* q, float* laneScales, Vector256<byte> scales, Vector256<float> acc)
    {
        var sums = Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(codes, Vector256.Load(q)), scales.AsInt16());
        return Accumulate(sums, Vector256.Load(laneScales), acc);
    }
}
