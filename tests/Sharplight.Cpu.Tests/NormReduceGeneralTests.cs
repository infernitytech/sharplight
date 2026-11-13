using Sharplight.Graph;

namespace Sharplight.Cpu.Tests;

public class NormReduceGeneralTests
{
    [Theory]
    [InlineData(1, 6, true, true), InlineData(2, 6, true, true), InlineData(6, 6, true, true), InlineData(3, 6, false, false), InlineData(2, 8, true, false)]
    public void GroupNorm(int groups, int C, bool scale, bool bias)
    {
        int N = 3;
        int H = 5;
        int W = 7;
        var x = Kit.Rand(N * C * H * W, 121, 2f);
        var sc = Kit.Rand(C, 122);
        var bi = Kit.Rand(C, 123);
        var tg = new TestGraph();
        var xin = tg.InF("x", SpatialKit.DSym(C, H, W));
        var ins = new List<Value> { xin };
        if (scale)
        {
            ins.Add(tg.WF("s", sc, C));
        }

        if (bias)
        {
            if (!scale)
            {
                ins.Add(tg.WF("s1", Enumerable.Repeat(1f, C).ToArray(), C));
            }
            ins.Add(tg.WF("b", bi, C));
        }
        tg.B.Output("y", tg.B.Emit1("GroupNorm", ins, GeneralKit.A(("groups", (long)groups), ("eps", 1e-5f))));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 3);
        Kit.Set(m, xin, x);
        var p = Kit.P(N);
        m.Run(p);
        var exp = new float[x.Length];
        int cg = C / groups;
        int S = H * W;
        for (int n = 0; n < N; n++)
        {
            for (int g = 0; g < groups; g++)
        {
            double mean = 0;
            double v = 0;
            long cnt = cg * S;
            for (int i = 0; i < cnt; i++)
                {
                    mean += x[(n * C + g * cg) * S + i];
                }

                mean /= cnt;
            for (int i = 0; i < cnt; i++)
                {
                    v += Math.Pow(x[(n * C + g * cg) * S + i] - mean, 2);
                }

                v /= cnt;
            for (int c = 0; c < cg; c++)
                {
                    for (int s = 0; s < S; s++)
            {
                int ch = g * cg + c;
                long o = (n * C + ch) * S + s;
                exp[o] = (float)((x[o] - mean) / Math.Sqrt(v + 1e-5) * (scale ? sc[ch] : 1) + (bias ? bi[ch] : 0));
            }
                }
            }
        }

        Kit.AssertClose(exp, Kit.Out(m, "y", p), 1e-4, 1e-4);
    }

    [Fact]
    public void GroupNormRank3()
    {
        int N = 2;
        int C = 4;
        int L = 9;
        var x = Kit.Rand(N * C * L, 124);
        var tg = new TestGraph();
        var xin = tg.InF("x", SpatialKit.DSym(C, L));
        tg.B.Output("y", tg.B.Emit1("GroupNorm", [xin], GeneralKit.A(("groups", 2L))));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 2);
        Kit.Set(m, xin, x);
        var p = Kit.P(N);
        m.Run(p);
        var exp = new float[x.Length];
        for (int n = 0; n < N; n++)
        {
            for (int g = 0; g < 2; g++)
        {
            int o0 = (n * C + g * 2) * L;
            int cnt = 2 * L;
            double mean = x.Skip(o0).Take(cnt).Average(v => (double)v);
            double var = x.Skip(o0).Take(cnt).Average(v => Math.Pow(v - mean, 2));
            for (int i = 0; i < cnt; i++)
                {
                    exp[o0 + i] = (float)((x[o0 + i] - mean) / Math.Sqrt(var + 1e-5));
                }
            }
        }

        Kit.AssertClose(exp, Kit.Out(m, "y", p), 1e-4, 1e-4);
    }

    public static IEnumerable<object[]> ReduceModes() { foreach (var mode in new[] { "prod", "l1", "l2", "sumsquare", "logsumexp" })
        {
            foreach (var (axes, keep, perm) in new (long[], bool, bool)[] { ([1], true, false), ([2, 3], false, false), ([0, 2], true, false), ([1, 2, 3], false, false), ([2, 3], true, true), ([1], false, true) })
            {
                yield return [mode, axes, keep, perm];
            }
        }
    }

    [Theory, MemberData(nameof(ReduceModes))]
    public void ReduceNewModes(string mode, long[] axes, bool keep, bool permuted)
    {
        long[] dims = [3, 4, 5, 6];
        long total = GeneralKit.P(dims);
        var x = Kit.Rand((int)total, 125, mode == "prod" ? 1.3f : 2f);
        var tg = new TestGraph();
        Value src;
        Value xin;
        float[] logical = x;
        if (permuted)
        {
            xin = tg.InF("x", SpatialKit.D(3, 5, 6, 4));
            src = tg.B.Permute(xin, 0, 3, 1, 2);
            logical = new float[total];
            for (int n = 0; n < 3; n++)
            {
                for (int h = 0; h < 5; h++)
                {
                    for (int w = 0; w < 6; w++)
                    {
                        for (int c = 0; c < 4; c++)
                        {
                            logical[((n * 4 + c) * 5 + h) * 6 + w] = x[((n * 5 + h) * 6 + w) * 4 + c];
                        }
                    }
                }
            }
        }
        else
        {
            xin = tg.InF("x", SpatialKit.D(dims));
            src = xin;
        }
        tg.B.Output("y", tg.B.Emit1("Reduce", [src], GeneralKit.A(("mode", mode), ("axes", axes), ("keepdims", keep))));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, xin, x);
        var p = Kit.P(1);
        m.Run(p);
        // reference through per-mode accumulation using Sum/Max reference of transformed data
        var red = new bool[4];
        foreach (var a in axes)
        {
            red[a] = true;
        }

        var keptDims = dims.Where((_, i) => !red[i]).ToArray();
        var exp = new float[GeneralKit.P(keptDims)];
        for (int o = 0; o < exp.Length; o++)
        {
            var kix = GeneralKit.Unravel(o, keptDims);
            var vals = new List<double>();
            for (long l = 0; l < total; l++)
            {
                var ix = GeneralKit.Unravel(l, dims);
                bool ok = true;
                for (int d = 0, q = 0; d < 4; d++)
                {
                    if (!red[d])
                    {
                        if (ix[d] != kix[q++])
                        {
                            ok = false;
                            break;
                        }
                    }
                }

                if (ok)
                {
                    vals.Add(logical[l]);
                }
            }
            exp[o] = (float)(mode switch
            {
                "prod" => vals.Aggregate(1.0, (a, b) => a * b), "l1" => vals.Sum(Math.Abs), "l2" => Math.Sqrt(vals.Sum(v => v * v)), "sumsquare" => vals.Sum(v => v * v),
                _ => Math.Log(vals.Sum(Math.Exp)),
            });
        }
        Kit.AssertClose(exp, Kit.Out(m, "y", p), 1e-4, 1e-4, mode);
        Assert.Equal(keep ? 4 : 4 - axes.Length, m.Graph.Outputs["y"].Shape.Rank);
    }

    [Fact]
    public void LogSoftmax()
    {
        int T = 3;
        int n = 19;
        var x = Kit.Rand(T * n, 126, 4f);
        var tg = new TestGraph();
        var xin = tg.InF("x", TestGraph.T, n);
        tg.B.Output("y", tg.B.Emit1("Softmax", [xin], GeneralKit.A(("log", true), ("scale", 0.8f))));
        tg.B.Output("y2", tg.B.Emit1("Softmax", [xin], GeneralKit.A(("log", true))));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, xin, x);
        var p = Kit.P(T);
        m.Run(p);
        var sm = Ref.Softmax(x, T, n, 0.8f);
        var sm2 = Ref.Softmax(x, T, n, 1f);
        Kit.AssertClose(sm.Select(v => (float)Math.Log(v)).ToArray(), Kit.Out(m, "y", p), 1e-5, 1e-5);
        Kit.AssertClose(sm2.Select(v => (float)Math.Log(v)).ToArray(), Kit.Out(m, "y2", p), 1e-5, 1e-5);
    }
}
