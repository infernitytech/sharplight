using System.Runtime.InteropServices;
using Sharplight.Tensors;

namespace Sharplight.Cpu.Tests;

public unsafe class QuantTests
{
    private record Fixture(DataType Type, int Rows, int Cols, float[] Src, byte[] Quant, float[] Deq);

    private static DataType FromGgml(int id) => id switch
    {
        2 => DataType.Q4_0, 3 => DataType.Q4_1, 6 => DataType.Q5_0, 7 => DataType.Q5_1, 8 => DataType.Q8_0,
        12 => DataType.Q4_K, 13 => DataType.Q5_K, 14 => DataType.Q6_K, _ => throw new NotSupportedException(id.ToString()),
    };

    private static List<Fixture> LoadFixtures()
    {
        var bytes = File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "Fixtures", "ggml_quant.bin"));
        var r = new BinaryReader(new MemoryStream(bytes));
        int n = r.ReadInt32();
        var list = new List<Fixture>();
        for (int i = 0; i < n; i++)
        {
            var type = FromGgml(r.ReadInt32());
            int rows = r.ReadInt32();
            int cols = r.ReadInt32();
            var src = new float[rows * cols];
            for (int k = 0; k < src.Length; k++)
            {
                src[k] = r.ReadSingle();
            }

            var q = r.ReadBytes((int)type.ByteSize((long)rows * cols));
            var dq = new float[rows * cols];
            for (int k = 0; k < dq.Length; k++)
            {
                dq[k] = r.ReadSingle();
            }

            list.Add(new Fixture(type, rows, cols, src, q, dq));
        }
        return list;
    }

    public static IEnumerable<object[]> FixtureNames() => LoadFixtures().Select(f => new object[] { f.Type.Name });

    /// <summary>Dequantization matches ggml's own dequantize_row_* (fixtures generated from libggml-base).</summary>
    [Theory, MemberData(nameof(FixtureNames))]
    public void DequantizeMatchesGgml(string typeName)
    {
        var f = LoadFixtures().First(x => x.Type.Name == typeName);
        var actual = new float[f.Rows * f.Cols];
        fixed (byte* q = f.Quant) fixed (float* d = actual)
        {
            Quant.Dequantize(Quant.FormatOf(f.Type)!.Value, q, actual.Length, d);
        }

        Kit.AssertClose(f.Deq, actual, 1e-5, 1e-5, typeName);
        // sanity: quantization error itself is moderate (guards against a broken fixture)
        double err = 0;
        double norm = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            err += Math.Pow(f.Src[i] - actual[i], 2);
            norm += f.Src[i] * f.Src[i];
        }
        Assert.True(err / norm < 0.05, $"{typeName} relative error {err / norm}");
    }

    /// <summary>Gather dequantizes rows (all supported weight formats, zero-copy and copied sources).</summary>
    [Theory, MemberData(nameof(FixtureNames))]
    public void GatherDequantizes(string typeName)
    {
        var f = LoadFixtures().First(x => x.Type.Name == typeName);
        foreach (bool zc in new[] { true, false })
        {
            var tg = new TestGraph();
            var table = tg.W("tbl", f.Type, new Shape(f.Rows, f.Cols), f.Quant, zc);
            var idx = tg.In("idx", DataType.I32, TestGraph.T);
            tg.B.Output("y", tg.B.Gather(table, idx));
            using var m = Kit.Compile(tg.B.Build());
            m.SetInput(idx, MemoryMarshal.AsBytes(new[] { 3, 0, 2, 2 }.AsSpan()));
            var p = Kit.P(4);
            m.Run(p);
            var y = Kit.Out(m, "y", p);
            var exp = new List<float>();
            foreach (int r in new[] { 3, 0, 2, 2 })
            {
                exp.AddRange(f.Deq.AsSpan(r * f.Cols, f.Cols).ToArray());
            }

            Kit.AssertClose(exp.ToArray(), y, 1e-5, 1e-5, typeName);
        }
    }

    [Fact]
    public void GatherOutOfRangeThrows()
    {
        var tg = new TestGraph();
        var table = tg.WF("tbl", Kit.Rand(4 * 8, 1), 4, 8);
        var idx = tg.In("idx", DataType.I32, TestGraph.T);
        tg.B.Output("y", tg.B.Gather(table, idx));
        using var m = Kit.Compile(tg.B.Build());
        m.SetInput(idx, MemoryMarshal.AsBytes(new[] { 4 }.AsSpan()));
        Assert.ThrowsAny<Exception>(() => m.Run(Kit.P(1)));
    }

    public static IEnumerable<object[]> MatMulCases()
    {
        foreach (var t in new[] { "f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "q4_k", "q5_k", "q6_k" })
        {
            foreach (var tokens in new[] { 1, 2, 5, 9, 37 })
            {
                yield return [t, tokens];
            }
        }
    }

    public static IEnumerable<object[]> IntDotMatMulCases()
    {
        foreach (var t in new[] { "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "q4_k", "q5_k", "q6_k" })
        {
            foreach (var tokens in new[] { 1, 3, 16, 40 })
            {
                yield return [t, tokens];
            }
        }
    }

    private static (byte[] data, float[] deq) MakeWeight(string type, int n, int k, int seed)
    {
        var w = Kit.Rand(n * k, seed, 0.5f);
        switch (type)
        {
            case "f32":
                return (MemoryMarshal.AsBytes(w.AsSpan()).ToArray(), w);
            case "f16":
            {
                var d = new byte[n * k * 2];
                var r = new float[n * k];
                for (int i = 0; i < w.Length; i++)
                {
                    var h = (Half)w[i];
                    BitConverter.TryWriteBytes(d.AsSpan(i * 2), h);
                    r[i] = (float)h;
                }
                return (d, r);
            }
            case "bf16":
            {
                var d = new byte[n * k * 2];
                var r = new float[n * k];
                for (int i = 0; i < w.Length; i++)
                {
                    uint bits = BitConverter.SingleToUInt32Bits(w[i]);
                    bits += 0x7FFF + ((bits >> 16) & 1);
                    ushort h = (ushort)(bits >> 16);
                    BitConverter.TryWriteBytes(d.AsSpan(i * 2), h);
                    r[i] = BitConverter.UInt32BitsToSingle((uint)h << 16);
                }
                return (d, r);
            }
            case "q8_0":
                {
                    var q = Quantizers.Q8_0(w);
                    return (q, Quantizers.DeqQ8_0(q));
                }
            case "q4_0":
                {
                    var q = Quantizers.Q4_0(w);
                    return (q, Quantizers.DeqQ4_0(q));
                }
            default:
            {
                // use the ggml fixture of the type: tile its rows to build an [n, k] weight (k must equal fixture cols multiple)
                var fx = LoadFixtures().First(x => x.Type.Name == type);
                int rowBytes = (int)fx.Type.ByteSize(fx.Cols);
                Assert.Equal(0, k % fx.Cols);
                var d = new byte[n * (int)fx.Type.ByteSize(k)];
                var r = new float[n * k];
                int perRow = k / fx.Cols;
                for (int i = 0; i < n; i++)
                    {
                        for (int c = 0; c < perRow; c++)
                    {
                        int src = (i + c) % fx.Rows;
                        Array.Copy(fx.Quant, src * rowBytes, d, (long)i * fx.Type.ByteSize(k) + c * rowBytes, rowBytes);
                        Array.Copy(fx.Deq, src * fx.Cols, r, i * k + c * fx.Cols, fx.Cols);
                    }
                    }

                    return (d, r);
            }
        }
    }

    /// <summary>
    /// MatMul over all formats; N not a multiple of 4, token counts for the row paths and the GEMM (partial M tiles); K spans
    /// several 1024-element blocks and GEMM K chunks.
    /// </summary>
    [Theory, MemberData(nameof(MatMulCases))]
    public void MatMulMatchesReference(string type, int tokens) => MatMulCheck(type, tokens, intDot: false);

    /// <summary>
    /// With integer dots enabled the activations are quantized to 8 bits per 32 values: the result stays within the
    /// quantization error (relative to the output norm) and the GEMM rows (40 tokens) remain exact.
    /// </summary>
    [Theory, MemberData(nameof(IntDotMatMulCases))]
    public void MatMulWithIntDotMatchesReference(string type, int tokens) => MatMulCheck(type, tokens, intDot: true);

    private static void MatMulCheck(string type, int tokens, bool intDot)
    {
        int n = 7;
        int k = type.EndsWith("_k") ? 2560 : 1536 + (type == "f32" ? 37 : 0);
        if (type is "q8_0" or "q4_0")
        {
            k = 1568; // 49 blocks
        }

        if (type is "q4_1" or "q5_0" or "q5_1")
        {
            k = 1024; // fixtures have 512 cols
        }

        var (data, deq) = MakeWeight(type, n, k, 7);
        var x = Kit.Rand(tokens * k, 11);
        var tg = new TestGraph();
        var w = tg.W("w", DataType.Get(type), new Shape(n, k), data);
        var xin = tg.InF("x", TestGraph.T, k);
        tg.B.Output("y", tg.B.MatMul(xin, w));
        using var m = Kit.Compile(tg.B.Build(), intDot: intDot);
        Kit.Set(m, xin, x);
        var p = Kit.P(tokens);
        m.Run(p);
        var y = Kit.Out(m, "y", p);
        var exp = new float[tokens * n];
        for (int t = 0; t < tokens; t++)
        {
            for (int r = 0; r < n; r++)
            {
                double s = 0;
                for (int c = 0; c < k; c++)
                {
                    s += (double)x[t * k + c] * deq[r * k + c];
                }

                exp[t * n + r] = (float)s;
            }
        }

        if (!intDot || tokens >= MatMulOp.IntDotGemmMinRows)
        {
            Kit.AssertClose(exp, y, 2e-3, 1e-4, type);
            return;
        }

        double err = 0;
        double norm = 0;
        for (int i = 0; i < exp.Length; i++)
        {
            err += Math.Pow(exp[i] - y[i], 2);
            norm += Math.Pow(exp[i], 2);
        }

        Assert.True(Math.Sqrt(err / norm) < 1e-2, $"{type} T={tokens}: relative error {Math.Sqrt(err / norm)}");
    }

    /// <summary>x is a strided view (Slice of a wider tensor) and the output feeds further ops.</summary>
    [Fact]
    public void MatMulOnStridedActivation()
    {
        int n = 5;
        int k = 64;
        int wide = 160;
        int off = 32;
        var wdata = Kit.Rand(n * k, 3);
        var tg = new TestGraph();
        var w = tg.WF("w", wdata, n, k);
        var xin = tg.InF("x", TestGraph.T, wide);
        var xs = tg.B.Slice(xin, 1, off, k);
        tg.B.Output("y", tg.B.MatMul(xs, w));
        using var m = Kit.Compile(tg.B.Build());
        var x = Kit.Rand(3 * wide, 4);
        Kit.Set(m, xin, x);
        var p = Kit.P(3);
        m.Run(p);
        var y = Kit.Out(m, "y", p);
        var exp = new float[3 * n];
        for (int t = 0; t < 3; t++)
        {
            for (int r = 0; r < n; r++)
            {
                double s = 0;
                for (int c = 0; c < k; c++)
                {
                    s += x[t * wide + off + c] * wdata[r * k + c];
                }
                exp[t * n + r] = (float)s;
            }
        }

        Kit.AssertClose(exp, y, 1e-4, 1e-4);
    }

    [Fact]
    public void OwnQuantizersRoundTrip()
    {
        var x = Kit.Rand(64 * 32, 5, 3f);
        var q8 = Quantizers.DeqQ8_0(Quantizers.Q8_0(x));
        var q4 = Quantizers.DeqQ4_0(Quantizers.Q4_0(x));
        double e8 = 0;
        double e4 = 0;
        double nn = 0;
        for (int i = 0; i < x.Length; i++)
        {
            e8 += Math.Pow(x[i] - q8[i], 2);
            e4 += Math.Pow(x[i] - q4[i], 2);
            nn += x[i] * x[i];
        }
        Assert.True(e8 / nn < 1e-4);
        Assert.True(e4 / nn < 0.02);
    }

    [Fact]
    public void UnsupportedWeightTypesAreReported()
    {
        var tg = new TestGraph();
        var w = tg.W("w", DataType.Q8_K, new Shape(4, 256), new byte[4 * 292]);
        var x = tg.InF("x", TestGraph.T, 256);
        tg.B.MatMul(x, w);
        var g2 = new TestGraph();
        var w2 = g2.W("w", DataType.Q8_K, new Shape(4, 256), new byte[4 * 292]);
        var x2 = g2.InF("x", TestGraph.T, 256);
        var y = g2.B.MatMul(x2, w2);
        g2.B.Output("y", y);
        var graph = g2.B.Build();
        var sup = Kit.Device.Supports(graph.Nodes[0]);
        Assert.False(sup.IsSupported);
        Assert.Contains("q8_k", sup.Reason!, StringComparison.OrdinalIgnoreCase);
    }
}
