using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

public sealed class RowOpTests
{
    [Theory]
    [InlineData(1, 100, 1f)]
    [InlineData(7, 1000, 0.125f)]
    [InlineData(5, 33, 1f)]
    [InlineData(3, 8, 2f)]
    [InlineData(2, 5000, 1f)]
    public void Softmax(int t, int d, float scale)
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", TG.T, d);
        tg.B.Output("y", tg.B.Emit1("Softmax", [x], Attributes.From(("scale", scale))));
        Diff.Check("Softmax", tg.B.Build(), new Step(t, 0, (x, Rnd.F((long)t * d, 3, 6f))), atol: 1e-6, rtol: 1e-4, maxTokens: 8);
    }

    [Fact]
    public void Softmax4DAndStrided()
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", TG.T, 4, 9, 20);
        var s = tg.B.Slice(x, 3, 2, 13);   // strided last axis slice (stride 20, inner contiguous)
        tg.B.Output("y", tg.B.Emit1("Softmax", [s], Attributes.Empty));
        Diff.Check("Softmax.4d", tg.B.Build(), new Step(3, 0, (x, Rnd.F(3 * 4 * 9 * 20, 3, 3f))), atol: 1e-6);
    }

    [Fact]
    public void SoftmaxOnPermutedInputFallsBackToDense()
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", TG.T, 6, 10);
        var p = tg.B.Permute(x, 0, 2, 1);   // inner axis no longer contiguous
        tg.B.Output("y", tg.B.Emit1("Softmax", [p], Attributes.Empty));
        Diff.Check("Softmax.permuted", tg.B.Build(), new Step(3, 0, (x, Rnd.F(3 * 60, 3, 3f))), atol: 1e-6);
    }

    [Theory]
    [InlineData(3, 64, true, true)]
    [InlineData(5, 768, true, true)]
    [InlineData(4, 100, false, false)]
    [InlineData(4, 100, true, false)]
    [InlineData(2, 33, false, true)]
    public void LayerNorm(int t, int d, bool w, bool b)
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", TG.T, 5, d);
        var args = new List<Value> { x };
        if (w)
        {
            args.Add(tg.RandW("w", DataType.F32, 1f, d));
        }

        if (b)
        {
            args.Add(tg.RandW("b", DataType.F32, 1f, d));
        }

        tg.B.Output("y", tg.B.Emit1("LayerNorm", args, Attributes.From(("eps", 1e-5f))));
        Diff.Check("LayerNorm", tg.B.Build(), new Step(t, 0, (x, Rnd.F((long)t * 5 * d, 3, 3f))), atol: 2e-5, rtol: 1e-4);
    }

    [Fact]
    public void LayerNormHalfWeights()
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", TG.T, 48);
        var w = tg.RandW("w", DataType.F16, 1f, 48);
        var b = tg.RandW("b", DataType.BF16, 1f, 48);
        tg.B.Output("y", tg.B.Emit1("LayerNorm", [x, w, b], Attributes.From(("eps", 1e-6f))));
        Diff.Check("LayerNorm.half", tg.B.Build(), new Step(4, 0, (x, Rnd.F(4 * 48, 3, 3f))), atol: 2e-5);
    }

    [Theory]
    [InlineData("ReduceSum")]
    [InlineData("ReduceMax")]
    [InlineData("ArgMax")]
    public void RowReduce(string op)
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", TG.T, 3, 77);
        tg.B.Output("y", tg.B.Emit1(op, [x], Attributes.Empty));
        Diff.Check(op, tg.B.Build(), new Step(5, 0, (x, Rnd.F(5 * 3 * 77, 3))), atol: 2e-4);
    }

    [Fact]
    public void ArgMaxTieBreaksToFirst()
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", TG.T, 40);
        tg.B.Output("y", tg.B.Emit1("ArgMax", [x], Attributes.Empty));
        var data = new float[2 * 40];
        data[7] = 5; data[33] = 5; data[40 + 39] = 2; data[40 + 1] = 2;
        Diff.Check("ArgMax.ties", tg.B.Build(), new Step(2, 0, (x, data)));
    }
}
