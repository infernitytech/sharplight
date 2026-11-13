using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Cpu.Tests;

/// <summary>Mamba-2 SelectiveScan (CPU) against a naive double-precision reference, plus CausalConv1d with bias.</summary>
public class SsmTests
{
    /// <summary>Naive recurrence; state [H, P, N] updated in place; x [T,H,P], dt [T,H], B/C [T,G,N].</summary>
    internal static float[] RefScan(float[] x, float[] dt, float[] dtBias, float[] A, float[] B, float[] C, float[] D, float[] S, int T, int H, int P, int G, int N, float dtMin)
    {
        var y = new float[T * H * P];
        for (int t = 0; t < T; t++)
            for (int h = 0; h < H; h++)
            {
                int g = h / (H / G);
                double v = dt[t * H + h] + dtBias[h];
                double d = Math.Max(v > 20 ? v : Math.Log(1 + Math.Exp(v)), dtMin);
                double decay = Math.Exp(d * A[h]);
                for (int p = 0; p < P; p++)
                {
                    double xp = x[(t * H + h) * P + p], acc = 0;
                    for (int n = 0; n < N; n++)
                    {
                        long si = ((long)h * P + p) * N + n;
                        double s = decay * S[si] + d * xp * B[(t * G + g) * N + n];
                        S[si] = (float)s;
                        acc += s * C[(t * G + g) * N + n];
                    }
                    y[(t * H + h) * P + p] = (float)(acc + D[h] * xp);
                }
            }
        return y;
    }

    [Theory]
    [InlineData(4, 8, 16, 2, false), InlineData(4, 8, 16, 2, true), InlineData(96, 80, 128, 8, true), InlineData(64, 64, 128, 8, false), InlineData(3, 5, 7, 3, true)]
    public void MatchesNaiveAndCarriesState(int H, int P, int N, int G, bool sliced)
    {
        int T = 9; float dtMin = 0.001f;
        // In the sliced variant x, B, C are views into one [T, H*P + 2*G*N] row like a conv output.
        int width = H * P + 2 * G * N;
        var row = Kit.Rand(T * width, 1);
        float[] x = new float[T * H * P], B = new float[T * G * N], C = new float[T * G * N];
        for (int t = 0; t < T; t++)
        {
            Array.Copy(row, t * width, x, t * H * P, H * P);
            Array.Copy(row, t * width + H * P, B, t * G * N, G * N);
            Array.Copy(row, t * width + H * P + G * N, C, t * G * N, G * N);
        }
        var dt = Kit.Rand(T * H, 2, 2f); var dtBias = Kit.Rand(H, 3, 1f);
        var A = Kit.Rand(H, 4).Select(v => -(0.1f + Math.Abs(v) * 4)).ToArray(); var D = Kit.Rand(H, 5);
        var S = new float[H * P * N];
        var expected = RefScan(x, dt, dtBias, A, B, C, D, S, T, H, P, G, N, dtMin);

        var tg = new TG();
        var xi = tg.InF("x", TG.T, sliced ? width : H * P); var dti = tg.InF("dt", TG.T, H);
        var bi = tg.InF("b", TG.T, G * N); var ci = tg.InF("c", TG.T, G * N);
        Value xv, bv, cv;
        if (sliced)
        {
            xv = tg.B.Reshape(tg.B.Slice(xi, 1, 0, H * P), 0, H, P);
            bv = tg.B.Reshape(tg.B.Slice(xi, 1, H * P, G * N), 0, G, N);
            cv = tg.B.Reshape(tg.B.Slice(xi, 1, H * P + G * N, G * N), 0, G, N);
        }
        else { xv = tg.B.Reshape(xi, 0, H, P); bv = tg.B.Reshape(bi, 0, G, N); cv = tg.B.Reshape(ci, 0, G, N); }
        var st = tg.State("ssm", DataType.F32, StateKind.Recurrent, H, P, N);
        tg.B.Output("y", tg.B.Emit("SelectiveScan", [xv, dti, tg.WF("dtb", dtBias, H), tg.WF("A", A, H), bv, cv, tg.WF("D", D, H), st],
            Attributes.From(("dt_min", dtMin)), writes: [st])[0]);
        var graph = tg.B.Build();

        foreach (var chunks in new[] { new[] { 9 }, [1, 1, 1, 1, 1, 1, 1, 1, 1], [4, 1, 4], [2, 7] })
        {
            using var m = Kit.Compile(graph, maxTokens: 16);
            var got = new List<float>(); int t0 = 0;
            foreach (int c in chunks)
            {
                Kit.Set(m, xi, sliced ? row[(t0 * width)..((t0 + c) * width)] : x[(t0 * H * P)..((t0 + c) * H * P)]);
                if (!sliced) { Kit.Set(m, bi, B[(t0 * G * N)..((t0 + c) * G * N)]); Kit.Set(m, ci, C[(t0 * G * N)..((t0 + c) * G * N)]); }
                Kit.Set(m, dti, dt[(t0 * H)..((t0 + c) * H)]);
                var p = Kit.P(c); m.Run(p); got.AddRange(Kit.Out(m, "y", p)); t0 += c;
            }
            Kit.AssertClose(expected, got.ToArray(), 1e-4, 1e-3, $"H{H} P{P} N{N} G{G} chunks {string.Join("+", chunks)}");
            var fin = ((CpuCompiledSegment)m.Segments[0]).TestGetState(st);
            Kit.AssertClose(S, fin, 1e-4, 1e-3, "final state");
        }
    }

    [Theory]
    [InlineData(20, 4, true), InlineData(20, 4, false), InlineData(600, 4, true)]
    public void CausalConvBias(int C, int K, bool silu)
    {
        int T = 6;
        var x = Kit.Rand(T * C, 1); var w = Kit.Rand(C * K, 2); var bias = Kit.Rand(C, 3);
        var tg = new TG(); var xin = tg.InF("x", TG.T, C);
        var state = tg.State("cs", DataType.F32, StateKind.Recurrent, K - 1, C);
        tg.B.Output("y", tg.B.Emit("CausalConv1d", [xin, tg.WF("w", w, C, K), state, tg.WF("b", bias, C)], Attributes.From(("activation", silu ? "silu" : "none")), writes: [state])[0]);
        var refNoAct = Ref.Conv(x, T, C, w, K, new float[(K - 1) * C], false);
        var exp = new float[T * C];
        for (int i = 0; i < exp.Length; i++) { float v = refNoAct[i] + bias[i % C]; exp[i] = silu ? v / (1f + MathF.Exp(-v)) : v; }
        var graph = tg.B.Build();
        foreach (var chunks in new[] { new[] { 6 }, [1, 2, 3] })
        {
            using var m = Kit.Compile(graph, maxTokens: 8);
            var got = new List<float>(); int t0 = 0;
            foreach (int c in chunks)
            {
                Kit.Set(m, xin, x[(t0 * C)..((t0 + c) * C)]);
                var p = Kit.P(c); m.Run(p); got.AddRange(Kit.Out(m, "y", p)); t0 += c;
            }
            Kit.AssertClose(exp, got.ToArray(), 1e-5, 1e-4, "conv bias");
        }
    }
}
