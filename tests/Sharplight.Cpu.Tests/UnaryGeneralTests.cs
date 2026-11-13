using Sharplight.Graph;

namespace Sharplight.Cpu.Tests;

public class UnaryGeneralTests
{
    private static readonly Dictionary<string, Func<double, double>> Ref = new()
    {
        ["Sin"] = Math.Sin, ["Cos"] = Math.Cos, ["Tan"] = Math.Tan, ["Asin"] = Math.Asin, ["Acos"] = Math.Acos, ["Atan"] = Math.Atan,
        ["Sinh"] = Math.Sinh, ["Cosh"] = Math.Cosh, ["Asinh"] = Math.Asinh, ["Acosh"] = Math.Acosh, ["Atanh"] = Math.Atanh,
        ["Floor"] = Math.Floor, ["Ceil"] = Math.Ceiling, ["Round"] = x => Math.Round(x, MidpointRounding.ToEven), ["Sign"] = x => Math.Sign(x),
        ["Softsign"] = x => x / (1 + Math.Abs(x)), ["Mish"] = x => x * Math.Tanh(Math.Log(1 + Math.Exp(x))),
        ["Elu"] = x => x >= 0 ? x : Math.Exp(x) - 1, ["Celu"] = x => Math.Max(0, x) + Math.Min(0, Math.Exp(x) - 1),
        ["Selu"] = x => x > 0 ? 1.0507009873554805 * x : 1.0507009873554805 * 1.6732632423543772 * (Math.Exp(x) - 1),
        ["LeakyRelu"] = x => x >= 0 ? x : 0.01 * x, ["ThresholdedRelu"] = x => x > 1 ? x : 0, ["Not"] = x => x != 0 ? 0 : 1,
    };

    public static IEnumerable<object[]> Names() => Ref.Keys.Select(k => new object[] { k });

    private static float[] Run(string op, float[] x, Attributes? attrs = null)
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

    [Theory, MemberData(nameof(Names))]
    public void Unary(string op)
    {
        var x = Kit.Rand(3 * 41, 91, op is "Asin" or "Acos" or "Atanh" ? 0.95f : 4f);
        if (op == "Acosh")
        {
            for (int i = 0; i < x.Length; i++)
            {
                x[i] = 1.01f + Math.Abs(x[i]) * 2;
            }
        }

        if (op == "Round")
        {
            x[0] = 0.5f;
            x[1] = 1.5f;
            x[2] = 2.5f;
            x[3] = -0.5f;
            x[4] = -1.5f;
            x[5] = -2.5f;
            x[6] = 2.4f;
            x[7] = 2.6f;
        }
        if (op == "Not")
        {
            x[0] = 0;
            x[1] = 2;
            x[2] = -3;
        }
        if (op == "Sign")
        {
            x[0] = 0;
        }

        Kit.AssertClose(x.Select(v => (float)Ref[op](v)).ToArray(), Run(op, x), 1e-5, 1e-5, op);
    }

    [Fact]
    public void AttributedActivations()
    {
        var x = Kit.Rand(3 * 41, 92, 3f);
        Kit.AssertClose(x.Select(v => v >= 0 ? v : 0.2f * v).ToArray(), Run("LeakyRelu", x, GeneralKit.A(("alpha", 0.2f))), 1e-6, 1e-6);
        Kit.AssertClose(x.Select(v => v > 0.5f ? v : 0f).ToArray(), Run("ThresholdedRelu", x, GeneralKit.A(("alpha", 0.5f))), 0, 0);
        Kit.AssertClose(x.Select(v => v >= 0 ? v : (float)(0.7 * (Math.Exp(v) - 1))).ToArray(), Run("Elu", x, GeneralKit.A(("alpha", 0.7f))), 1e-6, 1e-5);
        Kit.AssertClose(x.Select(v => (float)(Math.Max(0, v) + Math.Min(0, 0.7 * (Math.Exp(v / 0.7) - 1)))).ToArray(), Run("Celu", x, GeneralKit.A(("alpha", 0.7f))), 1e-6, 1e-5);
        Kit.AssertClose(x.Select(v => (float)(v > 0 ? 1.5 * v : 1.5 * 1.2 * (Math.Exp(v) - 1))).ToArray(), Run("Selu", x, GeneralKit.A(("alpha", 1.2f), ("gamma", 1.5f))), 1e-6, 1e-5);
    }

    [Fact]
    public void NanInfMasks()
    {
        var x = new float[] { float.NaN, 1f, float.PositiveInfinity, float.NegativeInfinity, 0f, -2f };
        var nan = new float[] { 1, 0, 0, 0, 0, 0 };
        var inf = new float[] { 0, 0, 1, 1, 0, 0 };
        var tg = new TestGraph();
        var xin = tg.InF("x", TestGraph.T, 3);
        tg.B.Output("nan", tg.B.Emit1("IsNaN", [xin]));
        tg.B.Output("inf", tg.B.Emit1("IsInf", [xin]));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, xin, x);
        var p = Kit.P(2);
        m.Run(p);
        Kit.AssertClose(nan, Kit.Out(m, "nan", p), 0, 0);
        Kit.AssertClose(inf, Kit.Out(m, "inf", p), 0, 0);
    }
}
