using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

public sealed class NormTests
{
    [Theory]
    [InlineData("RmsNorm", 1, 64, true)] [InlineData("RmsNorm", 6, 2048, true)] [InlineData("RmsNorm", 5, 100, false)] [InlineData("RmsNorm", 3, 33, true)]
    [InlineData("L2Norm", 1, 64, false)] [InlineData("L2Norm", 6, 128, false)] [InlineData("L2Norm", 4, 50, false)]
    public void Dense(string op, int t, int d, bool weight)
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", TG.T, d);
        var args = new List<Value> { x };
        if (weight)
        {
            args.Add(tg.RandW("w", DataType.F32, 1f, d));
        }

        tg.B.Output("y", tg.B.Emit1(op, args, Attributes.From(("eps", 1e-6f))));
        Diff.Check(op, tg.B.Build(), new Step(t, 0, (x, Rnd.F((long)t * d, 3))), atol: 2e-5, rtol: 1e-4);
    }

    [Theory]
    [InlineData("RmsNorm", true)] [InlineData("RmsNorm", false)] [InlineData("L2Norm", false)]
    public void ThreeAndFourDimStrided(string op, bool weight)
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", TG.T, 6, 48);
        var s = tg.B.Slice(x, 2, 8, 32);                        // [T,6,32], row stride 48
        var args = new List<Value> { s };
        if (weight)
        {
            args.Add(tg.RandW("w", DataType.F32, 1f, 32));
        }

        tg.B.Output("y", tg.B.Emit1(op, args, Attributes.From(("eps", 1e-5f))));
        var x4 = tg.InF("x4", TG.T, 3, 5, 24);
        tg.B.Output("y4", tg.B.Emit1(op, [x4], Attributes.From(("eps", 1e-5f))));
        var p = tg.B.Permute(x4, 0, 2, 1, 3);                   // [T,5,3,24], middle dims not collapsible -> dense temp
        tg.B.Output("yp", tg.B.Emit1(op, [p], Attributes.From(("eps", 1e-5f))));
        Diff.Check(op + ".strided", tg.B.Build(), new Step(5, 0, (x, Rnd.F(5 * 288, 3)), (x4, Rnd.F(5 * 360, 4))), atol: 2e-5, rtol: 1e-4);
    }
}
