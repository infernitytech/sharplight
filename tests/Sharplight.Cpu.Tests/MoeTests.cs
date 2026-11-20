using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Cpu.Tests;

public class MoeTests
{
    /// <summary>MatMulId against a naive per-slot reference, shared-x [T,K] and per-slot-x [T,S,K], f32 and f16 expert weights.</summary>
    [Theory]
    [InlineData(false, false, 5, false), InlineData(true, false, 5, false), InlineData(false, true, 5, false), InlineData(true, true, 5, false)]
    [InlineData(false, false, 1, false), InlineData(false, true, 37, false), InlineData(true, false, 37, false), InlineData(false, false, 37, true), InlineData(true, true, 60, true)]
    public void MatMulIdMatchesReference(bool perSlot, bool f16, int T, bool sparse)
    {
        int E = 6, N = 22, K = 64, S = 3;   // experts 0, 1, 3, 5 get no rows when sparse; N is no multiple of 4
        var rng = new Random(3);
        var wf = Enumerable.Range(0, E * N * K).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
        if (f16)
        {
            for (int i = 0; i < wf.Length; i++)
            {
                wf[i] = (float)(Half)wf[i];
            }
        }

        var x = Kit.Rand(perSlot ? T * S * K : T * K, 7);
        var ids = Enumerable.Range(0, T * S).Select(i => sparse ? (i % 3 == 0 ? 2 : 4) : (i * 5 + i / 3) % E).ToArray();

        var tg = new TestGraph();
        var xin = perSlot ? tg.InF("x", TestGraph.T, S, K) : tg.InF("x", TestGraph.T, K);
        var idin = tg.In("ids", DataType.I32, TestGraph.T, S);
        Value w;
        if (f16)
        {
            var bytes = new byte[wf.Length * 2];
            for (int i = 0; i < wf.Length; i++)
            {
                BitConverter.TryWriteBytes(bytes.AsSpan(i * 2), (Half)wf[i]);
            }

            w = tg.W("w", DataType.F16, new Shape(E, N, K), bytes);
        }
        else
        {
            w = tg.WF("w", wf, E, N, K);
        }

        tg.B.Output("y", tg.B.Emit1("MatMulId", [xin, w, idin]));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, xin, x);
        Kit.Set(m, idin, ids);
        var p = Kit.P(T);
        m.Run(p);

        var exp = new float[T * S * N];
        for (int t = 0; t < T; t++)
        {
            for (int s = 0; s < S; s++)
            {
                for (int n = 0; n < N; n++)
                {
                    double acc = 0;
                    int e = ids[t * S + s];
                    for (int k = 0; k < K; k++)
                    {
                        acc += (double)wf[(e * N + n) * K + k] * x[(perSlot ? (t * S + s) : t) * K + k];
                    }

                    exp[(t * S + s) * N + n] = (float)acc;
                }
            }
        }

        Kit.AssertClose(exp, Kit.Out(m, "y", p), 1e-4, 1e-4);
    }
    /// <summary>The ggml-quantized fixture records of every format the integer-dot MatMulId path decodes, with an even row count.</summary>
    public static IEnumerable<object[]> IntDotRecords()
    {
        var records = GgmlDequantTests.Load();
        for (int i = 0; i < records.Count; i++)
        {
            if (Quant.FormatOf(records[i].Type) is { } f && IntDot.Supports(f) && records[i].Rows >= 2 && records[i].Rows % 2 == 0
                && records.FindIndex(r => r.Type == records[i].Type) == i)
            {
                yield return [i, records[i].Type.Name];
            }
        }
    }

    /// <summary>
    /// MatMulId with integer dots on quantized experts (two experts from a fixture's rows) against the dequantized weights in double
    /// precision, within the error of 8-bit activations (experts with more than four items take the float path for the grid formats); a
    /// one-row run gives the first row of a two-row run bit for bit.
    /// </summary>
    [Theory, MemberData(nameof(IntDotRecords))]
    public void QuantizedMatMulIdWithIntegerDotsMatchesTheDequantizedWeights(int index, string label)
    {
        var record = GgmlDequantTests.Load()[index];
        int e = 2, n = record.Rows / 2, k = record.Cols, s = 2, t = 5;
        var x = Kit.Rand(t * k, 11);
        var ids = Enumerable.Range(0, t * s).Select(i => (i * 7 + i / 2) % e).ToArray();
        float[] Run(int rows)
        {
            var tg = new TestGraph();
            var xin = tg.InF("x", TestGraph.T, k);
            var idin = tg.In("ids", DataType.I32, TestGraph.T, s);
            var w = tg.W("w", record.Type, new Shape(e, n, k), record.Blocks);
            tg.B.Output("y", tg.B.Emit1("MatMulId", [xin, w, idin]));
            using var m = Kit.Compile(tg.B.Build(), intDot: true);
            Kit.Set(m, xin, x[..(rows * k)]);
            Kit.Set(m, idin, ids[..(rows * s)]);
            var p = Kit.P(rows);
            m.Run(p);
            return Kit.Out(m, "y", p);
        }

        var all = Run(t);
        for (int row = 0; row < t; row++)
        {
            for (int slot = 0; slot < s; slot++)
            {
                for (int j = 0; j < n; j++)
                {
                    double acc = 0;
                    double magnitude = 0;
                    for (int c = 0; c < k; c++)
                    {
                        double product = (double)record.Expected[(ids[row * s + slot] * n + j) * k + c] * x[row * k + c];
                        acc += product;
                        magnitude += Math.Abs(product);
                    }

                    float got = all[(row * s + slot) * n + j];
                    Assert.True(Math.Abs(got - acc) <= 0.02 * magnitude + 1e-6, $"{label} [{row}, {slot}, {j}]: {got} vs {acc} (magnitude {magnitude})");
                }
            }
        }

        // Up to four items per expert every item is computed alone: one row gives the first row of a two-row run bit for bit.
        Assert.Equal(Run(2)[..(s * n)], Run(1));
    }
}
