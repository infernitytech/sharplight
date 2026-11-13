using Sharplight.Graph;

namespace Sharplight.Cpu.Tests;

public class NewElementwiseTests
{
    private static float[] Run1(string op, float[] x, Attributes? attrs = null)
    {
        var tg = new TestGraph();
        var xin = tg.InF("x", TestGraph.T, x.Length / 3);
        tg.B.Output("y", tg.B.Emit1(op, [xin], attrs));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, xin, x);
        var p = Kit.P(3);
        m.Run(p);
        return Kit.Out(m, "y", p);
    }

    [Fact]
    public void ErfIsAccurate()
    {
        int n = 3 * 4001;
        var x = new float[n];
        for (int i = 0; i < n; i++)
        {
            x[i] = -8f + 16f * i / (n - 1);
        }

        var y = Run1("Erf", x);
        double worst = 0;
        float wx = 0;
        for (int i = 0; i < n; i++)
        {
            double e = Math.Abs(y[i] - ScalarMath.Erf(x[i]));
            if (e > worst)
            {
                worst = e;
                wx = x[i];
            }
        }
        Assert.True(worst < 2e-7, $"max abs erf error {worst:E3} at x={wx}");
    }

    [Fact]
    public void ErfOddLengthTailAndSpecials()
    {
        var x = new float[] { 0f, 1e-8f, -1e-3f, 0.5f, -0.5f, 3.5f, -3.5f, 10f, -10f };
        var y = Run1("Erf", x);
        for (int i = 0; i < x.Length; i++)
        {
            Assert.True(Math.Abs(y[i] - ScalarMath.Erf(x[i])) < 2e-7, $"erf({x[i]}) = {y[i]}");
        }
    }

    [Fact]
    public void HardSwishAndSigmoid()
    {
        var x = Kit.Rand(3 * 37, 71, 8f);
        Kit.AssertClose(x.Select(v => (float)(v * Math.Clamp(v / 6.0 + 0.5, 0, 1))).ToArray(), Run1("HardSwish", x), 1e-6, 1e-6);
        Kit.AssertClose(x.Select(v => (float)Math.Clamp(0.2 * v + 0.5, 0, 1)).ToArray(), Run1("HardSigmoid", x), 1e-6, 1e-6);
        Kit.AssertClose(x.Select(v => (float)Math.Clamp(Math.Round(1.0 / 6, 10) * v + 0.5, 0, 1)).ToArray(),
            Run1("HardSigmoid", x, Attributes.From(("alpha", 1f / 6f), ("beta", 0.5f))), 1e-6, 1e-6);
        Kit.AssertClose(x.Select(v => (float)Math.Clamp(0.1 * v - 0.3, 0, 1)).ToArray(),
            Run1("HardSigmoid", x, Attributes.From(("alpha", 0.1f), ("beta", -0.3f))), 1e-6, 1e-6);
    }

    [Fact]
    public void ClipVariants()
    {
        var x = Kit.Rand(3 * 37, 72, 4f);
        Kit.AssertClose(x.Select(v => Math.Min(Math.Max(v, -1f), 2.5f)).ToArray(), Run1("Clip", x, Attributes.From(("min", -1f), ("max", 2.5f))), 0, 0);
        Kit.AssertClose(x.Select(v => Math.Max(v, 0f)).ToArray(), Run1("Clip", x, Attributes.From(("min", 0f))), 0, 0);
        Kit.AssertClose(x.Select(v => Math.Min(v, 6f)).ToArray(), Run1("Clip", x, Attributes.From(("max", 6f))), 0, 0);
        Kit.AssertClose(x.Select(v => Math.Clamp(v, 0f, 6f)).ToArray(), Run1("Clip", x, Attributes.From(("min", 0f), ("max", 6f))), 0, 0);
    }

    [Theory]
    [InlineData(2f), InlineData(0.5f), InlineData(3f), InlineData(1f), InlineData(-1.5f), InlineData(0f)]
    public void PowScalarExponent(float e)
    {
        var x = Kit.Rand(3 * 37, 73).Select(v => Math.Abs(v) + 0.1f).ToArray();
        var tg = new TestGraph();
        var xin = tg.InF("x", TestGraph.T, 37);
        tg.B.Output("y", tg.B.Binary("Pow", xin, tg.WF("e", [e], 1)));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, xin, x);
        var p = Kit.P(3);
        m.Run(p);
        Kit.AssertClose(x.Select(v => (float)Math.Pow(v, e)).ToArray(), Kit.Out(m, "y", p), 1e-6, 1e-5);
    }

    [Fact]
    public void Nchw4dBroadcastConstants()
    {
        int N = 3;
        int C = 6;
        int H = 5;
        int W = 7;
        var x = Kit.Rand(N * C * H * W, 74);
        var sc = Kit.Rand(C, 75);
        var bi = Kit.Rand(C, 76);
        var tg = new TestGraph();
        var xin = tg.InF("x", SpatialKit.DSym(C, H, W));
        var scv = tg.WF("sc", sc, C, 1, 1);
        var biv = tg.WF("bi", bi, C, 1, 1);
        var y = tg.B.Add(tg.B.Mul(xin, scv), biv);
        tg.B.Output("y", y);
        tg.B.Output("pw", tg.B.Binary("Pow", tg.B.Unary("Abs", xin), tg.WF("pe", [1.7f], 1, 1, 1)));
        tg.B.Output("sub", tg.B.Sub(xin, tg.WF("c4", Kit.Rand(C, 77), 1, C, 1, 1)));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 3);
        Kit.Set(m, xin, x);
        var p = Kit.P(N);
        m.Run(p);
        var exp = new float[x.Length];
        var pw = new float[x.Length];
        for (int i = 0; i < x.Length; i++)
        {
            int c = i / (H * W) % C;
            exp[i] = x[i] * sc[c] + bi[c];
            pw[i] = (float)Math.Pow(Math.Abs(x[i]), 1.7f);
        }
        Kit.AssertClose(exp, Kit.Out(m, "y", p), 1e-6, 1e-5);
        Kit.AssertClose(pw, Kit.Out(m, "pw", p), 1e-6, 1e-5);
    }

    [Fact]
    public void SoftmaxAndLayerNormOnHigherRanks()
    {
        int N = 3;
        // 4-D softmax over the last axis, 3-D layer norm with weight/bias
        var x4 = Kit.Rand(N * 2 * 3 * 17, 78);
        var x3 = Kit.Rand(N * 5 * 24, 79);
        var w = Kit.Rand(24, 80);
        var b = Kit.Rand(24, 81);
        var tg = new TestGraph();
        var i4 = tg.InF("x4", SpatialKit.DSym(2, 3, 17));
        var i3 = tg.InF("x3", SpatialKit.DSym(5, 24));
        tg.B.Output("sm", tg.B.Emit1("Softmax", [i4]));
        tg.B.Output("ln", tg.B.Emit1("LayerNorm", [i3, tg.WF("w", w, 24), tg.WF("b", b, 24)], Attributes.From(("eps", 1e-5f))));
        tg.B.Output("rms", tg.B.RmsNorm(tg.B.Permute(i4, 0, 1, 3, 2), null, 1e-6f));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 3);
        Kit.Set(m, i4, x4);
        Kit.Set(m, i3, x3);
        var p = Kit.P(N);
        m.Run(p);
        Kit.AssertClose(Ref.Softmax(x4, N * 2 * 3, 17, 1f), Kit.Out(m, "sm", p), 1e-6, 1e-5);
        var ln = new float[x3.Length];
        for (int r = 0; r < N * 5; r++)
        {
            double mean = 0;
            for (int i = 0; i < 24; i++)
            {
                mean += x3[r * 24 + i];
            }

            mean /= 24;
            double v = 0;
            for (int i = 0; i < 24; i++)
            {
                v += Math.Pow(x3[r * 24 + i] - mean, 2);
            }

            v /= 24;
            for (int i = 0; i < 24; i++)
            {
                ln[r * 24 + i] = (float)((x3[r * 24 + i] - mean) / Math.Sqrt(v + 1e-5) * w[i] + b[i]);
            }
        }
        Kit.AssertClose(ln, Kit.Out(m, "ln", p), 1e-5, 1e-4);
        var perm = new float[x4.Length];
        for (int r = 0; r < N * 2; r++)
        {
            for (int a = 0; a < 3; a++)
            {
                for (int c = 0; c < 17; c++)
                {
                    perm[(r * 17 + c) * 3 + a] = x4[(r * 3 + a) * 17 + c];
                }
            }
        }

        Kit.AssertClose(Ref.RmsNorm(perm, N * 2 * 17, 3, null, 1e-6f), Kit.Out(m, "rms", p), 1e-5, 1e-4);
    }

    [Fact]
    public void GeluUsesAccurateErf()
    {
        var x = Kit.Rand(3 * 101, 82, 5f);
        var y = Run1("Gelu", x.Concat(new float[3 * 101 - x.Length]).ToArray());
        for (int i = 0; i < x.Length; i++)
        {
            Assert.True(Math.Abs(y[i] - 0.5 * x[i] * (1 + ScalarMath.Erf(x[i] / Math.Sqrt(2)))) < 2e-6);
        }
    }
}
