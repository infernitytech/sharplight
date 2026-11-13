namespace Sharplight.Cpu.Tests;

public class BinaryGeneralTests
{
    private static float Fmod(float a, float b, bool fmod)
    {
        float r = (float)Math.IEEERemainder(0, 1);
        r = a % b;
        if (!fmod && r != 0 && (r < 0) != (b < 0))
        {
            r += b;
        }
        return r;
    }

    [Theory]
    [InlineData("Mod", true), InlineData("Mod", false), InlineData("PRelu", false), InlineData("Equal", false), InlineData("NotEqual", false), InlineData("Less", false),
     InlineData("LessEqual", false), InlineData("Greater", false), InlineData("GreaterEqual", false), InlineData("And", false), InlineData("Or", false), InlineData("Xor", false)]
    public void Binary(string op, bool fmod)
    {
        int T = 3;
        int n = 20;
        var a = Kit.Rand(T * n, 93, 5f);
        var b = Kit.Rand(n, 94, 3f);
        // make some values equal / zero / integral so comparisons and masks have ties
        for (int i = 0; i < b.Length; i += 3)
        {
            b[i] = a[i];
        }

        for (int i = 0; i < a.Length; i += 4)
        {
            a[i] = 0;
        }

        for (int i = 1; i < b.Length; i += 5)
        {
            b[i] = 0;
        }

        if (op == "Mod")
        {
            for (int i = 0; i < b.Length; i++)
            {
                if (b[i] == 0)
                {
                    b[i] = 1.7f;
                }
            }
        }

        var tg = new TestGraph();
        var ain = tg.InF("a", TestGraph.T, n);
        var bv = tg.WF("b", b, n);
        tg.B.Output("y", tg.B.Emit1(op, [ain, bv], op == "Mod" ? GeneralKit.A(("fmod", fmod)) : null));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, ain, a);
        var p = Kit.P(T);
        m.Run(p);
        var exp = new float[T * n];
        for (int i = 0; i < exp.Length; i++)
        {
            float u = a[i];
            float v = b[i % n];
            exp[i] = op switch
            {
                "Mod" => Fmod(u, v, fmod), "PRelu" => u < 0 ? u * v : u, "Equal" => u == v ? 1 : 0, "NotEqual" => u != v ? 1 : 0, "Less" => u < v ? 1 : 0,
                "LessEqual" => u <= v ? 1 : 0, "Greater" => u > v ? 1 : 0, "GreaterEqual" => u >= v ? 1 : 0, "And" => u != 0 && v != 0 ? 1 : 0,
                "Or" => u != 0 || v != 0 ? 1 : 0, _ => (u != 0) != (v != 0) ? 1 : 0,
            };
        }
        Kit.AssertClose(exp, Kit.Out(m, "y", p), 1e-6, 1e-6, op);
    }

    [Fact]
    public void ComparisonBroadcastAndPReluSlopePerChannel()
    {
        int N = 2;
        int C = 3;
        int H = 4;
        int W = 5;
        var x = Kit.Rand(N * C * H * W, 95);
        var slope = Kit.Rand(C, 96);
        var tg = new TestGraph();
        var xin = tg.InF("x", SpatialKit.DSym(C, H, W));
        tg.B.Output("p", tg.B.Emit1("PRelu", [xin, tg.WF("s", slope, C, 1, 1)]));
        tg.B.Output("g", tg.B.Emit1("Greater", [xin, tg.WF("z", [0f], 1, 1, 1)]));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 2);
        Kit.Set(m, xin, x);
        var p = Kit.P(N);
        m.Run(p);
        Kit.AssertClose(x.Select((v, i) => v < 0 ? v * slope[i / (H * W) % C] : v).ToArray(), Kit.Out(m, "p", p), 1e-6, 1e-6);
        Kit.AssertClose(x.Select(v => v > 0 ? 1f : 0f).ToArray(), Kit.Out(m, "g", p), 0, 0);
    }
}
