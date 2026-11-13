using System.Runtime.InteropServices;

namespace Sharplight.Cpu.Tests;

/// <summary>
/// Integer-dot kernels (<see cref="IntDot"/>): weight decoders against ggml's dequantization, activation quantization against a
/// scalar reference, the scalar dot against a double-precision product of the decoded operands, and every instruction-set tier
/// the host supports against the scalar kernel (the int32 sums are exact; only the float scaling order differs).
/// </summary>
public unsafe class IntDotTests
{
    private static readonly WeightFormat[] s_formats =
    [
        WeightFormat.Q8_0, WeightFormat.Q4_0, WeightFormat.Q4_1, WeightFormat.Q5_0, WeightFormat.Q5_1, WeightFormat.Q4_K,
        WeightFormat.Q5_K, WeightFormat.Q6_K, WeightFormat.IQ4_NL, WeightFormat.IQ4_XS, WeightFormat.MXFP4,
    ];

    /// <summary>The ggml-quantized fixture records of every format with an integer-dot decoder.</summary>
    public static IEnumerable<object[]> DecoderRecords()
    {
        var records = GgmlDequantTests.Load();
        for (int i = 0; i < records.Count; i += 2)
        {
            var f = Quant.FormatOf(records[i].Type);
            if (f is { } format && s_formats.Contains(format))
            {
                yield return [i, records[i].Type.Name];
            }
        }
    }

    /// <summary>Every supported tier above scalar.</summary>
    public static IEnumerable<object[]> Tiers() =>
        CpuIsa.Supported.Where(l => l != CpuIsaLevel.Scalar).Select(l => new object[] { CpuIsa.Name(l) });

    [Fact]
    public void FormatsAreSupported()
    {
        foreach (var f in s_formats)
        {
            Assert.True(IntDot.Supports(f), f.ToString());
        }
    }

    /// <summary>a·u + b of the decoded row equals ggml's dequantized values.</summary>
    [Theory, MemberData(nameof(DecoderRecords))]
    public void DecodeMatchesDequantize(int index, string label)
    {
        var record = GgmlDequantTests.Load()[index];
        var format = Quant.FormatOf(record.Type)!.Value;
        int k = record.Cols;
        long rowBytes = record.Type.ByteSize(k);
        var u = new byte[k];
        var a = new float[k / IntDot.GroupSize];
        var b = new float[k / IntDot.GroupSize];
        for (int r = 0; r < record.Rows; r++)
        {
            fixed (byte* src = record.Blocks, pu = u)
            fixed (float* pa = a, pb = b)
            {
                IntDot.Decode(format, src + r * rowBytes, k, pu, pa, pb);
            }

            for (int i = 0; i < k; i++)
            {
                if (IntDot.HasSmallCodes(format))
                {
                    Assert.True(u[i] < 128, $"{label} row {r} code {i} = {u[i]} is not small");
                }

                float expected = record.Expected[r * k + i];
                float actual = a[i / IntDot.GroupSize] * u[i] + b[i / IntDot.GroupSize];
                Assert.True(Math.Abs(expected - actual) <= 1e-6 * Math.Abs(expected) + 1e-7, $"{label} row {r} [{i}]: {expected} vs {actual}");
            }
        }
    }

    /// <summary>Per block of 32: scale max|x| / 127, round half away from zero, and the scaled group sums.</summary>
    [Fact]
    public void QuantizeMatchesReference()
    {
        const int k = 32 * 9;
        var x = Kit.Rand(k, 3, 4f);
        for (int i = 32; i < 64; i++)
        {
            x[i] = 0f; // an all-zero block
        }

        for (int i = 64; i < 96; i++)
        {
            x[i] = (i - 80) * 0.5f; // exact halves after scaling by 127 / 8
        }

        x[100] = 1e-30f;
        x[130] = -1e30f;
        using var row = new Row(k);
        fixed (float* px = x)
        {
            IntDot.Quantize(px, k, row.Q8);
        }

        for (int blk = 0; blk < k / 32; blk++)
        {
            float amax = 0f;
            for (int i = 0; i < 32; i++)
            {
                amax = MathF.Max(amax, MathF.Abs(x[blk * 32 + i]));
            }

            float d = amax / 127f;
            float id = d != 0f ? 1f / d : 0f;
            int[] sums = new int[2];
            for (int i = 0; i < 32; i++)
            {
                float v = x[blk * 32 + i] * id;
                int q = (int)(v + MathF.CopySign(0.5f, v));
                Assert.Equal(q, row.Q8.Q[blk * 32 + i]);
                sums[i / 16] += q;
            }

            for (int l = 0; l < 8; l++)
            {
                Assert.Equal(d, row.Q8.LaneScales[blk * 8 + l]);
            }

            Assert.Equal(d * sums[0], row.Q8.GroupSums[blk * 2]);
            Assert.Equal(d * sums[1], row.Q8.GroupSums[blk * 2 + 1]);
        }
    }

    /// <summary>The scalar kernel equals Σ (a·u + b)(d·q) in double precision.</summary>
    [Fact]
    public void ScalarDotMatchesDoubleReference()
    {
        foreach (bool smallCodes in new[] { true, false })
        {
            var c = Case.Random(1024, smallCodes, 5);
            float got = c.Dot(CpuIsaLevel.Scalar);
            double expected = 0;
            double magnitude = 0;
            for (int i = 0; i < c.K; i++)
            {
                double w = (double)c.A[i / 16] * c.U[i] + c.B[i / 16];
                double xv = (double)c.Row.Q8.LaneScales[i / 4] * c.Row.Q8.Q[i];
                expected += w * xv;
                magnitude += Math.Abs(w * xv);
            }

            Assert.True(Math.Abs(got - expected) <= 1e-5 * magnitude, $"small={smallCodes}: {got} vs {expected} (magnitude {magnitude})");
            c.Dispose();
        }
    }

    /// <summary>Each tier against the scalar kernel on random codes, saturating extremes and real decoded rows.</summary>
    [Theory, MemberData(nameof(Tiers))]
    public void TierMatchesScalar(string tier)
    {
        var level = CpuIsa.Supported.First(l => CpuIsa.Name(l) == tier);
        var cases = new List<Case>();
        foreach (int k in new[] { 32, 96, 256, 512 + 32, 4096 })
        {
            cases.Add(Case.Random(k, smallCodes: true, seed: k));
            cases.Add(Case.Random(k, smallCodes: false, seed: k + 1));
        }

        // vpmaddubsw saturation: largest codes against the largest activations of both signs
        cases.Add(Case.Constant(512, code: 255, activation: 1f));
        cases.Add(Case.Constant(512, code: 255, activation: -1f));
        cases.Add(Case.Constant(512, code: 128, activation: -1f));
        cases.Add(Case.Constant(512, code: 127, activation: 1f, smallCodes: true));
        var records = GgmlDequantTests.Load();
        foreach (var args in DecoderRecords())
        {
            cases.Add(Case.FromRecord(records[(int)args[0]]));
        }

        foreach (var c in cases)
        {
            float expected = c.Dot(CpuIsaLevel.Scalar);
            float actual = c.Dot(level);
            double tolerance = 2e-6 * c.Magnitude() + 1e-6;
            Assert.True(Math.Abs(expected - actual) <= tolerance, $"{tier} {c.Label}: {actual} vs scalar {expected}");
            c.Dispose();
        }
    }

    /// <summary>An aligned activation row buffer.</summary>
    private sealed class Row : IDisposable
    {
        private readonly byte* _p;

        public Row(int k)
        {
            long bytes = IntDot.RowBytes(k);
            _p = (byte*)NativeMemory.AlignedAlloc((nuint)bytes, 64);
            NativeMemory.Clear(_p, (nuint)bytes);
            Q8 = IntDot.RowAt(_p, k);
        }

        public Q8Row Q8 { get; }

        public void Dispose() => NativeMemory.AlignedFree(_p);
    }

    /// <summary>A decoded weight row and a quantized activation row.</summary>
    private sealed class Case : IDisposable
    {
        private Case(string label, int k, byte[] u, float[] a, float[] b, bool smallCodes, float[] x)
        {
            Label = label;
            K = k;
            U = u;
            A = a;
            B = b;
            SmallCodes = smallCodes;
            Row = new Row(k);
            fixed (float* px = x)
            {
                IntDot.Quantize(px, k, Row.Q8);
            }
        }

        public string Label { get; }

        public int K { get; }

        public byte[] U { get; }

        public float[] A { get; }

        public float[] B { get; }

        public bool SmallCodes { get; }

        public Row Row { get; }

        public static Case Random(int k, bool smallCodes, int seed)
        {
            var rng = new Random(seed);
            var u = new byte[k];
            for (int i = 0; i < k; i++)
            {
                u[i] = (byte)rng.Next(smallCodes ? 128 : 256);
            }

            var a = Kit.Rand(k / 16, seed + 100, 0.01f);
            var b = Kit.Rand(k / 16, seed + 200, 0.5f);
            return new Case($"random k={k} small={smallCodes}", k, u, a, b, smallCodes, Kit.Rand(k, seed + 300, 2f));
        }

        public static Case Constant(int k, byte code, float activation, bool smallCodes = false)
        {
            var u = Enumerable.Repeat(code, k).ToArray();
            var a = Enumerable.Repeat(0.01f, k / 16).ToArray();
            var b = Enumerable.Repeat(-0.3f, k / 16).ToArray();
            return new Case($"constant code={code} x={activation}", k, u, a, b, smallCodes, Enumerable.Repeat(activation, k).ToArray());
        }

        public static Case FromRecord(GgmlDequantTests.Record record)
        {
            var format = Quant.FormatOf(record.Type)!.Value;
            int k = record.Cols;
            var u = new byte[k];
            var a = new float[k / 16];
            var b = new float[k / 16];
            fixed (byte* src = record.Blocks, pu = u)
            fixed (float* pa = a, pb = b)
            {
                IntDot.Decode(format, src, k, pu, pa, pb);
            }

            return new Case($"{record.Type.Name} row 0", k, u, a, b, IntDot.HasSmallCodes(format), Kit.Rand(k, 17));
        }

        public float Dot(CpuIsaLevel level)
        {
            fixed (byte* pu = U)
            fixed (float* pa = A, pb = B)
            {
                return IntDot.Dot(level, pu, pa, pb, SmallCodes, Row.Q8, K);
            }
        }

        /// <summary>Σ |a·u·d·q| + |b·d·q|, the scale of the float rounding error.</summary>
        public double Magnitude()
        {
            double m = 0;
            for (int i = 0; i < K; i++)
            {
                double xv = Math.Abs((double)Row.Q8.LaneScales[i / 4] * Row.Q8.Q[i]);
                m += (Math.Abs((double)A[i / 16] * U[i]) + Math.Abs(B[i / 16])) * xv;
            }

            return m;
        }

        public void Dispose() => Row.Dispose();
    }
}
