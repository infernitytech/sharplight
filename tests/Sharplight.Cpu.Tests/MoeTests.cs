using Sharplight.Graph;
using Sharplight.Tensors;
using Xunit;

namespace Sharplight.Cpu.Tests;

public class MoeTests
{
    /// <summary>MatMulId against a naive per-slot reference, shared-x [T,K] and per-slot-x [T,S,K], f32 and f16 expert weights.</summary>
    [Theory]
    [InlineData(false, false), InlineData(true, false), InlineData(false, true), InlineData(true, true)]
    public void MatMulIdMatchesReference(bool perSlot, bool f16)
    {
        int E = 6, N = 20, K = 64, S = 3, T = 5;
        var rng = new Random(3);
        var wf = Enumerable.Range(0, E * N * K).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
        if (f16) for (int i = 0; i < wf.Length; i++) wf[i] = (float)(Half)wf[i];
        var x = Kit.Rand(perSlot ? T * S * K : T * K, 7);
        var ids = Enumerable.Range(0, T * S).Select(i => (i * 5 + i / 3) % E).ToArray();

        var tg = new TG();
        var xin = perSlot ? tg.InF("x", TG.T, S, K) : tg.InF("x", TG.T, K);
        var idin = tg.In("ids", DataType.I32, TG.T, S);
        Value w;
        if (f16)
        {
            var bytes = new byte[wf.Length * 2];
            for (int i = 0; i < wf.Length; i++) BitConverter.TryWriteBytes(bytes.AsSpan(i * 2), (Half)wf[i]);
            w = tg.W("w", DataType.F16, new Shape(E, N, K), bytes);
        }
        else w = tg.WF("w", wf, E, N, K);
        tg.B.Output("y", tg.B.Emit1("MatMulId", [xin, w, idin]));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, xin, x); Kit.Set(m, idin, ids);
        var p = Kit.P(T); m.Run(p);

        var exp = new float[T * S * N];
        for (int t = 0; t < T; t++)
            for (int s = 0; s < S; s++)
                for (int n = 0; n < N; n++)
                {
                    double acc = 0;
                    int e = ids[t * S + s];
                    for (int k = 0; k < K; k++) acc += (double)wf[(e * N + n) * K + k] * x[(perSlot ? (t * S + s) : t) * K + k];
                    exp[(t * S + s) * N + n] = (float)acc;
                }
        Kit.AssertClose(exp, Kit.Out(m, "y", p), 1e-4, 1e-4);
    }
}
