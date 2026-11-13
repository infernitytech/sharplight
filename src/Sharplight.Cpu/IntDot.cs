using System.Runtime.Intrinsics;

namespace Sharplight.Cpu;

/// <summary>
/// Integer-dot matrix-vector products for quantized weights (<see cref="Runtime.DeviceFeatures.IntDot"/>), the llama.cpp
/// approach: the activations are quantized to 8 bits per 32 values (<see cref="Q8Row"/>), a weight row is decoded into
/// unsigned 8-bit codes <c>u</c> with one multiplier <c>a</c> and one offset <c>b</c> per 16 values (value = a * u + b), and
/// the products u * q are summed exactly in int32 lanes of four (VNNI <c>vpdpbusd</c>, AVX2 <c>vpmaddubsw</c>, 128-bit
/// widening multiplies or scalar code). Per 16 values:
/// <c>sum(w * x) = a * dx * sum(u * q) + b * dx * sum(q)</c>.
/// </summary>
/// <remarks>
/// Every <see cref="CpuIsaLevel"/> has its own kernel (<see cref="Dot(CpuIsaLevel, byte*, float*, float*, bool, Q8Row, int)"/>);
/// the integer part is exact on every tier, only the float accumulation order differs.
/// </remarks>
internal static unsafe partial class IntDot
{
    /// <summary>Values per activation scale.</summary>
    public const int BlockSize = 32;

    /// <summary>Values per weight multiplier / offset.</summary>
    public const int GroupSize = 16;

    /// <summary>Bytes of one quantized activation row of <paramref name="k"/> values (values, lane scales, group sums).</summary>
    public static long RowBytes(long k) => Align(k) + Align(k) + Align(k / GroupSize * 4);

    /// <summary>The row layout of <see cref="RowBytes"/> at <paramref name="p"/>.</summary>
    public static Q8Row RowAt(byte* p, long k) => new((sbyte*)p, (float*)(p + Align(k)), (float*)(p + 2 * Align(k)));

    /// <summary>Bytes of a decoded weight row of <paramref name="k"/> values (codes, multipliers, offsets).</summary>
    public static long DecodedBytes(long k) => Align(k) + 2 * Align(k / GroupSize * 4);

    /// <summary>Whether <paramref name="f"/> has a decoder for the integer-dot path.</summary>
    public static bool Supports(WeightFormat f) => f is WeightFormat.Q8_0 or WeightFormat.Q4_0 or WeightFormat.Q4_1 or WeightFormat.Q5_0
        or WeightFormat.Q5_1 or WeightFormat.Q4_K or WeightFormat.Q5_K or WeightFormat.Q6_K or WeightFormat.IQ4_NL or WeightFormat.IQ4_XS
        or WeightFormat.MXFP4;

    /// <summary>
    /// Whether the decoded codes of <paramref name="f"/> stay below 128, so <c>vpmaddubsw</c> cannot saturate and needs no
    /// sign handling (8-bit and codebook formats use the full 0..255 range).
    /// </summary>
    public static bool HasSmallCodes(WeightFormat f) => f is not (WeightFormat.Q8_0 or WeightFormat.IQ4_NL or WeightFormat.IQ4_XS);

    /// <summary>
    /// Quantizes <paramref name="k"/> (a multiple of 32) values: per block of 32 the scale is max|x| / 127 and the values
    /// round half away from zero. The vector and scalar paths give identical results.
    /// </summary>
    public static void Quantize(float* x, long k, Q8Row dst)
    {
        for (long b = 0; b < k; b += BlockSize)
        {
            if (CpuIsa.Use256)
            {
                QuantizeBlock256(x + b, dst.Q + b, dst.LaneScales + b / 4, dst.GroupSums + b / GroupSize);
            }
            else
            {
                QuantizeBlockScalar(x + b, dst.Q + b, dst.LaneScales + b / 4, dst.GroupSums + b / GroupSize);
            }
        }
    }

    /// <summary>Dot product of a decoded weight row with one quantized activation row on the given tier.</summary>
    /// <param name="level">Kernel tier; must be supported by the host (<see cref="CpuIsa.IsSupported"/>).</param>
    /// <param name="u">Weight codes, <paramref name="k"/> bytes.</param>
    /// <param name="a">Weight multiplier per 16 values.</param>
    /// <param name="b">Weight offset per 16 values.</param>
    /// <param name="smallCodes">Every code is below 128 (<see cref="HasSmallCodes"/>).</param>
    /// <param name="x">The activation row.</param>
    /// <param name="k">Row length, a multiple of 32.</param>
    public static float Dot(CpuIsaLevel level, byte* u, float* a, float* b, bool smallCodes, Q8Row x, int k) => level switch
    {
        CpuIsaLevel.Avx512Vnni => DotAvx512Vnni(u, a, b, x, k),
        CpuIsaLevel.Avx512 => DotAvx512(u, a, b, smallCodes, x, k),
        CpuIsaLevel.AvxVnni => DotAvxVnni(u, a, b, x, k),
        CpuIsaLevel.Avx2 => DotAvx2(u, a, b, smallCodes, x, k),
        CpuIsaLevel.Vector128 => DotVector128(u, a, b, x, k),
        _ => DotScalar(u, a, b, x, k),
    };

    private static long Align(long bytes) => (bytes + 63) & ~63L;

    private static void QuantizeBlockScalar(float* x, sbyte* q, float* laneScales, float* groupSums)
    {
        float amax = 0f;
        for (int i = 0; i < BlockSize; i++)
        {
            amax = MathF.Max(amax, MathF.Abs(x[i]));
        }

        float d = amax / 127f;
        float id = d != 0f ? 1f / d : 0f;
        int s0 = 0;
        int s1 = 0;
        for (int i = 0; i < BlockSize; i++)
        {
            float v = x[i] * id;
            int r = (int)(v + MathF.CopySign(0.5f, v));
            q[i] = (sbyte)r;
            if (i < GroupSize)
            {
                s0 += r;
            }
            else
            {
                s1 += r;
            }
        }

        for (int l = 0; l < BlockSize / 4; l++)
        {
            laneScales[l] = d;
        }

        groupSums[0] = d * s0;
        groupSums[1] = d * s1;
    }

    private static void QuantizeBlock256(float* x, sbyte* q, float* laneScales, float* groupSums)
    {
        var v0 = Vector256.Load(x);
        var v1 = Vector256.Load(x + 8);
        var v2 = Vector256.Load(x + 16);
        var v3 = Vector256.Load(x + 24);
        var m = Vector256.Max(Vector256.Max(Vector256.Abs(v0), Vector256.Abs(v1)), Vector256.Max(Vector256.Abs(v2), Vector256.Abs(v3)));
        m = Vector256.Max(m, Vector256.Shuffle(m, Vector256.Create(4, 5, 6, 7, 0, 1, 2, 3)));
        m = Vector256.Max(m, Vector256.Shuffle(m, Vector256.Create(2, 3, 0, 1, 6, 7, 4, 5)));
        m = Vector256.Max(m, Vector256.Shuffle(m, Vector256.Create(1, 0, 3, 2, 5, 4, 7, 6)));
        float amax = m.ToScalar();
        float d = amax / 127f;
        float id = d != 0f ? 1f / d : 0f;
        var idv = Vector256.Create(id);
        var half = Vector256.Create(0.5f);
        var i0 = Round(v0 * idv, half);
        var i1 = Round(v1 * idv, half);
        var i2 = Round(v2 * idv, half);
        var i3 = Round(v3 * idv, half);
        var lo = Vector256.Narrow(i0, i1);
        var hi = Vector256.Narrow(i2, i3);
        Vector256.Narrow(lo, hi).Store(q);
        Vector256.Create(d).Store(laneScales);
        groupSums[0] = d * Vector256.Sum(i0 + i1);
        groupSums[1] = d * Vector256.Sum(i2 + i3);
    }

    private static Vector256<int> Round(Vector256<float> v, Vector256<float> half) =>
        Vector256.ConvertToInt32(v + Vector256.CopySign(half, v));
}
