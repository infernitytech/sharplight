using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

public sealed class ElementwiseTests
{
    private static readonly string[] Positive = ["Log", "Sqrt", "Rsqrt", "Recip"];

    public static IEnumerable<object[]> UnaryOps() => StandardOps.UnaryNames.Select(n => new object[] { n });

    [Theory]
    [MemberData(nameof(UnaryOps))]
    public void Unary(string op)
    {
        Gpu.Require();
        int T = 5;
        int D = 37;
        var tg = new TG();
        var x = tg.InF("x", TG.T, D);
        tg.B.Output("y", tg.B.Unary(op, x));
        var data = Positive.Contains(op) ? Rnd.Pos(T * D, 1) : Rnd.F(T * D, 1, 4f);
        if (op is "Asin" or "Acos" or "Atanh")
        {
            data = Rnd.F(T * D, 1, 0.95f);
        }

        if (op == "Acosh")
        {
            data = Rnd.Pos(T * D, 1, 1.05f, 6f);
        }

        if (op is "Sinh" or "Cosh")
        {
            data = Rnd.F(T * D, 1, 3f);
        }

        if (op is "Round" or "Floor" or "Ceil" or "Sign")
        {
            data = Rnd.F(T * D, 1, 6f);
            data[0] = 0.5f;
            data[1] = 1.5f;
            data[2] = 2.5f;
            data[3] = -0.5f;
            data[4] = -1.5f;
            data[5] = -2.5f;
            data[6] = 0f;
            data[7] = 3f;
        }
        if (op is "Not")
        {
            data = [.. data.Select((v, i) => i % 3 == 0 ? 0f : v)];
        }

        if (op is "IsNaN" or "IsInf")
        {
            data[0] = float.NaN;
            data[1] = float.PositiveInfinity;
            data[2] = float.NegativeInfinity;
            data[3] = 0f;
        }
        if (op == "Exp")
        {
            data = Rnd.F(T * D, 1, 3f);
        }

        Diff.Check("Unary." + op, tg.B.Build(), new Step(T, 0, (x, data)), atol: 2e-5, rtol: 2e-5);
    }

    [Fact]
    public void ErfAccuracyAcrossRange()
    {
        Gpu.Require();
        int n = 4001;
        var data = Enumerable.Range(0, n).Select(i => -6f + 12f * i / (n - 1)).ToArray();
        var tg = new TG();
        var x = tg.InF("x", TG.T, 1);
        tg.B.Output("y", tg.B.Unary("Erf", x));
        // the CPU oracle is a double-precision erf in a perfect world; compare tightly
        Diff.Check("Erf.range", tg.B.Build(), new Step(n, 0, (x, data)), atol: 5e-7, rtol: 5e-7, maxTokens: n);
    }

    [Theory]
    [InlineData(0.2f, 0.5f), InlineData(1f / 6f, 0.5f), InlineData(0.3f, 0.1f)]
    public void HardSigmoidAttrs(float alpha, float beta)
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", TG.T, 33);
        tg.B.Output("y", tg.B.Emit1("HardSigmoid", [x], Attributes.From(("alpha", alpha), ("beta", beta))));
        Diff.Check("Unary.HardSigmoid.attrs", tg.B.Build(), new Step(6, 0, (x, Rnd.F(6 * 33, 3, 8f))));
    }

    [Theory]
    [InlineData(-1f, 1f), InlineData(0f, 6f), InlineData(float.NegativeInfinity, 0.5f), InlineData(-0.5f, float.PositiveInfinity)]
    public void Clip(float lo, float hi)
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", TG.T, 4, 9);
        var attrs = Attributes.Empty;
        if (!float.IsNegativeInfinity(lo))
        {
            attrs = attrs.With("min", lo);
        }

        if (!float.IsPositiveInfinity(hi))
        {
            attrs = attrs.With("max", hi);
        }

        tg.B.Output("y", tg.B.Emit1("Clip", [x], attrs));
        Diff.Check("Clip", tg.B.Build(), new Step(7, 0, (x, Rnd.F(7 * 36, 4, 3f))));
    }

    [Fact]
    public void PowPositiveBaseAndFractionalExponent()
    {
        Gpu.Require();
        var tg = new TG();
        var a = tg.InF("a", TG.T, 20);
        var b = tg.InF("b", TG.T, 20);
        tg.B.Output("y", tg.B.Binary("Pow", a, b));
        Diff.Check("Pow.pos", tg.B.Build(), new Step(4, 0, (a, Rnd.Pos(80, 1, 0.1f, 5f)), (b, Rnd.F(80, 2, 3f))), atol: 1e-4, rtol: 1e-4);
    }

    [Fact]
    public void PowNegativeBaseIntegerExponent()
    {
        Gpu.Require();
        var tg = new TG();
        var a = tg.InF("a", TG.T, 8);
        var b = tg.InF("b", TG.T, 8);
        tg.B.Output("y", tg.B.Binary("Pow", a, b));
        var av = new float[] { -2, -2, -2, -2, 0, 0, 3, 2.5f, -1.5f, -3, 4, 0, 1, -1, 2, 7 };
        var bv = new float[] { 0, 1, 2, 3, 0, 2, 0, 2, 3, 3, 0.5f, 1, 5, 4, -1, 2 };
        Diff.Check("Pow.neg", tg.B.Build(), new Step(2, 0, (a, av), (b, bv)), atol: 1e-4, rtol: 1e-5);
    }

    [Theory]
    [InlineData("Add"), InlineData("Sub"), InlineData("Mul"), InlineData("Div"), InlineData("Max"), InlineData("Min")]
    public void BinaryBroadcast(string op)
    {
        Gpu.Require();
        var tg = new TG();
        var a = tg.InF("a", TG.T, 6, 5, 4);
        var b = tg.InF("b", 1, 6, 1, 4);
        tg.B.Output("y", tg.B.Binary(op, a, b));
        Diff.Check("Binary." + op, tg.B.Build(), new Step(3, 0, (a, Rnd.F(3 * 120, 1)), (b, Rnd.Pos(24, 2))));
    }

    [Theory]
    [InlineData("Equal")] [InlineData("NotEqual")] [InlineData("Less")] [InlineData("LessEqual")] [InlineData("Greater")] [InlineData("GreaterEqual")]
    [InlineData("And")] [InlineData("Or")] [InlineData("Xor")] [InlineData("PRelu")]
    public void ComparisonLogicAndPRelu(string op)
    {
        Gpu.Require();
        var tg = new TG();
        var a = tg.InF("a", TG.T, 5, 7);
        var b = tg.InF("b", 1, 5, 1);
        tg.B.Output("y", tg.B.Binary(op, a, b));
        var av = Rnd.F(4 * 35, 1, 2f).Select(v => MathF.Round(v)).ToArray();   // plenty of exact ties and zeros
        var bv = Rnd.F(5, 2, 2f).Select(v => MathF.Round(v)).ToArray();
        Diff.Check("Binary." + op, tg.B.Build(), new Step(4, 0, (a, av), (b, bv)));
    }

    [Theory] [InlineData(true)] [InlineData(false)]
    public void Mod(bool fmod)
    {
        Gpu.Require();
        var tg = new TG();
        var a = tg.InF("a", TG.T, 16);
        var b = tg.InF("b", TG.T, 16);
        tg.B.Output("y", tg.B.Emit1("Mod", [a, b], Attributes.From(("fmod", fmod))));
        var bv = Rnd.F(48, 2, 4f).Select(v => v == 0 ? 1f : v).ToArray();
        Diff.Check("Binary.Mod", tg.B.Build(), new Step(3, 0, (a, Rnd.F(48, 1, 9f)), (b, bv)), atol: 1e-5, what: $"fmod={fmod}");
    }

    [Theory]
    [InlineData("Elu", 0.7f)] [InlineData("Celu", 1.3f)] [InlineData("Selu", 1.2f)] [InlineData("LeakyRelu", 0.1f)] [InlineData("ThresholdedRelu", 0.5f)]
    public void ParameterizedActivations(string op, float alpha)
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", TG.T, 40);
        tg.B.Output("y", tg.B.Emit1(op, [x], Attributes.From(("alpha", alpha))));
        Diff.Check("Unary." + op + ".alpha", tg.B.Build(), new Step(3, 0, (x, Rnd.F(120, 1, 3f))), atol: 2e-5, rtol: 2e-5);
    }

    [Fact]
    public void WhereFused()
    {
        Gpu.Require();
        var tg = new TG();
        var c = tg.InF("c", TG.T, 1, 9);
        var a = tg.InF("a", TG.T, 6, 9);
        var b = tg.InF("b", 6, 1);
        tg.B.Output("y", tg.B.Unary("Relu", tg.B.Emit1("Where", [tg.B.Binary("Greater", c, tg.B.Scale(a, 0f, 0f)), a, b])));
        Diff.Check("Where.f32", tg.B.Build(), new Step(2, 0, (c, Rnd.F(18, 1)), (a, Rnd.F(108, 2)), (b, Rnd.F(6, 3))));
    }

    [Fact]
    public void Nchw4DConstantBroadcast()
    {
        Gpu.Require();
        int C = 16;
        int H = 7;
        int W = 9;
        int N = 2;
        var tg = new TG();
        var x = tg.InF("x", TG.T, C, H, W);
        var scale = tg.Const("scale", Rnd.Pos(C, 1), C, 1, 1);
        var bias = tg.RandW("bias", DataType.F32, 1f, 1, C, 1, 1);
        var y = tg.B.Add(tg.B.Mul(x, scale), bias);
        y = tg.B.Unary("Relu", y);
        tg.B.Output("y", y);
        Diff.Check("Elementwise.nchw", tg.B.Build(), new Step(N, 0, (x, Rnd.F(N * C * H * W, 2))));
    }

    [Theory]
    [InlineData("f16"), InlineData("bf16")]
    public void HalfPrecisionOperand(string t)
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", TG.T, 24);
        var w = tg.RandW("w", DataType.Get(t), 1f, 24);
        tg.B.Output("y", tg.B.Mul(tg.B.Add(x, w), tg.B.Silu(x)));
        Diff.Check("Elementwise.half." + t, tg.B.Build(), new Step(5, 0, (x, Rnd.F(120, 3))));
    }

    [Fact]
    public void FusedChainWithMultipleLiveOutputs()
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", TG.T, 50);
        var a = tg.B.Silu(x);
        var b = tg.B.Mul(a, tg.B.Scale(x, 0.5f, 1f));
        var c = tg.B.Add(b, tg.B.Unary("Tanh", a));
        tg.B.Output("a", a); tg.B.Output("c", c);
        Diff.Check("Elementwise.fused", tg.B.Build(), new Step(9, 0, (x, Rnd.F(450, 5, 2f))));
    }

    [Fact]
    public void StridedInputs()
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", TG.T, 8, 12);
        var s = tg.B.Slice(x, 2, 3, 6);                        // [T,8,6] strided
        var p = tg.B.Permute(s, 0, 2, 1);                      // [T,6,8]
        tg.B.Output("y", tg.B.Add(tg.B.Unary("Exp", p), tg.B.Scale(p, 2f)));
        Diff.Check("Elementwise.strided", tg.B.Build(), new Step(4, 0, (x, Rnd.F(4 * 96, 6))));
    }
}
