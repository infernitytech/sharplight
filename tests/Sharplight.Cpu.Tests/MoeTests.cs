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
}
