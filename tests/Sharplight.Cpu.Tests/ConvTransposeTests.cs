using Sharplight.Graph;

namespace Sharplight.Cpu.Tests;

public class ConvTransposeTests
{
    public record Case(string Name, int C, int H, int W, int O, int KH, int KW, int[] S, int[] P, int[] D, int[] OP, int G, bool Bias);
    private static readonly Case[] Cases =
    [
        new("k2s2", 4, 5, 6, 3, 2, 2, [2, 2], [0, 0, 0, 0], [1, 1], [0, 0], 1, true),
        new("k3s2p1op1", 5, 6, 7, 4, 3, 3, [2, 2], [1, 1, 1, 1], [1, 1], [1, 1], 1, true),
        new("k4s2p1", 6, 8, 8, 5, 4, 4, [2, 2], [1, 1, 1, 1], [1, 1], [0, 0], 1, false),
        new("k3s1p1", 3, 9, 10, 7, 3, 3, [1, 1], [1, 1, 1, 1], [1, 1], [0, 0], 1, true),
        new("k3s1", 3, 4, 5, 2, 3, 3, [1, 1], [0, 0, 0, 0], [1, 1], [0, 0], 1, false),
        new("dil2", 4, 6, 6, 3, 3, 3, [1, 1], [2, 2, 2, 2], [2, 2], [0, 0], 1, true),
        new("groups2", 6, 5, 5, 8, 3, 3, [2, 2], [1, 1, 1, 1], [1, 1], [1, 1], 2, true),
        new("asym", 4, 5, 7, 6, 3, 2, [2, 1], [1, 0, 0, 1], [1, 1], [0, 0], 2, true),
        new("longK", 300, 3, 4, 5, 3, 3, [2, 2], [1, 1, 1, 1], [1, 1], [1, 1], 1, true),
        new("rect-s3", 4, 4, 5, 3, 3, 3, [3, 3], [0, 0, 0, 0], [1, 1], [0, 0], 1, true),
    ];
    public static IEnumerable<object[]> Params() { foreach (var c in Cases)
        {
            foreach (var n in new[] { 1, 3 })
            {
                yield return [c.Name, n];
            }
        }
    }

    [Theory, MemberData(nameof(Params))]
    public void ConvTranspose(string name, int N)
    {
        var c = Cases.Single(k => k.Name == name);
        int Og = c.O / c.G;
        int Cg = c.C / c.G;
        var x = Kit.Rand(N * c.C * c.H * c.W, 131);
        var w = Kit.Rand(c.C * Og * c.KH * c.KW, 132, 0.5f);
        var bias = c.Bias ? Kit.Rand(c.O, 133) : null;
        var tg = new TestGraph();
        var xin = tg.InF("x", SpatialKit.DSym(c.C, c.H, c.W));
        var ins = new List<Value> { xin, tg.WF("w", w, c.C, Og, c.KH, c.KW) };
        if (bias != null)
        {
            ins.Add(tg.WF("b", bias, c.O));
        }

        long[] L(int[] v) => v.Select(q => (long)q).ToArray();
        tg.B.Output("y", tg.B.Emit1("ConvTranspose", ins, GeneralKit.A(("groups", (long)c.G), ("strides", L(c.S)), ("pads", L(c.P)), ("dilations", L(c.D)), ("output_padding", L(c.OP)))));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 3);
        Kit.Set(m, xin, x);
        var p = Kit.P(N);
        m.Run(p);
        int HO = (c.H - 1) * c.S[0] - c.P[0] - c.P[2] + c.D[0] * (c.KH - 1) + c.OP[0] + 1;
        int WO = (c.W - 1) * c.S[1] - c.P[1] - c.P[3] + c.D[1] * (c.KW - 1) + c.OP[1] + 1;
        var exp = new double[N * c.O * HO * WO];
        for (int n = 0; n < N; n++)
        {
            for (int o = 0; o < c.O; o++)
            {
                for (int i = 0; i < HO * WO; i++)
                {
                    exp[(n * c.O + o) * HO * WO + i] = bias?[o] ?? 0;
                }
            }
        }

        for (int n = 0; n < N; n++)
        {
            for (int g = 0; g < c.G; g++)
            {
                for (int cc = 0; cc < Cg; cc++)
                {
                    for (int o = 0; o < Og; o++)
                    {
                        for (int ih = 0; ih < c.H; ih++)
                        {
                            for (int iw = 0; iw < c.W; iw++)
            {
                double xv = x[((n * c.C + g * Cg + cc) * c.H + ih) * c.W + iw];
                for (int kh = 0; kh < c.KH; kh++)
                                {
                                    for (int kw = 0; kw < c.KW; kw++)
                {
                    int oh = ih * c.S[0] - c.P[0] + kh * c.D[0];
                    int ow = iw * c.S[1] - c.P[1] + kw * c.D[1];
                    if (oh < 0 || oh >= HO || ow < 0 || ow >= WO)
                                        {
                                            continue;
                                        }

                                        exp[((n * c.O + g * Og + o) * HO + oh) * WO + ow] += xv * w[(((g * Cg + cc) * Og + o) * c.KH + kh) * c.KW + kw];
                }
                                }
                            }
                        }
                    }
                }
            }
        }

        var shape = m.Graph.Outputs["y"].Shape;
        Assert.Equal(HO, (int)shape[2].Fixed);
        Assert.Equal(WO, (int)shape[3].Fixed);
        Kit.AssertClose(exp.Select(v => (float)v).ToArray(), Kit.Out(m, "y", p), 1e-4, 1e-4, name);
    }

    [Fact]
    public void ConvTransposeF16Weights()
    {
        int C = 4;
        int O = 3;
        int H = 5;
        int W = 5;
        var x = Kit.Rand(C * H * W, 134);
        var wf = Kit.Rand(C * O * 4, 135, 0.5f);
        var tg = new TestGraph();
        var xin = tg.InF("x", SpatialKit.DSym(C, H, W));
        var wv = SpatialKit.Weight(tg, "w", wf, [C, O, 2, 2], "f16", out var weff);
        tg.B.Output("y", tg.B.Emit1("ConvTranspose", [xin, wv], GeneralKit.A(("strides", new long[] { 2, 2 }))));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 1);
        Kit.Set(m, xin, x);
        var p = Kit.P(1);
        m.Run(p);
        var exp = new float[O * 10 * 10];
        for (int c = 0; c < C; c++)
        {
            for (int o = 0; o < O; o++)
            {
                for (int ih = 0; ih < H; ih++)
                {
                    for (int iw = 0; iw < W; iw++)
                    {
                        for (int kh = 0; kh < 2; kh++)
                        {
                            for (int kw = 0; kw < 2; kw++)
                            {
                                exp[(o * 10 + ih * 2 + kh) * 10 + iw * 2 + kw] += x[(c * H + ih) * W + iw] * weff[((c * O + o) * 2 + kh) * 2 + kw];
                            }
                        }
                    }
                }
            }
        }

        Kit.AssertClose(exp, Kit.Out(m, "y", p), 1e-4, 1e-4);
    }
}
