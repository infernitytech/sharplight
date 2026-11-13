using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

public sealed class ViewTests
{
    [Fact]
    public void SliceConcatAndContiguous()
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", TG.T, 4, 10);
        var a = tg.B.Slice(x, 2, 0, 4);
        var b = tg.B.Slice(x, 2, 4, 6);
        tg.B.Output("cat", tg.B.Concat(2, b, a));
        tg.B.Output("cat1", tg.B.Concat(1, tg.B.Slice(x, 1, 0, 1), tg.B.Slice(x, 1, 2, 2)));
        tg.B.Output("contig", tg.B.Contiguous(tg.B.Permute(x, 0, 2, 1)));
        Diff.Check("Views.concat", tg.B.Build(), new Step(5, 0, (x, Rnd.F(5 * 40, 1))), atol: 0, rtol: 0);
    }

    [Fact]
    public void ReshapeOfNonContiguousViewFallsBackToCopy()
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", TG.T, 6, 8);
        var p = tg.B.Permute(x, 0, 2, 1);              // [T,8,6] strided
        var r = tg.B.Reshape(p, 0, 48);                // cannot be a view of the permuted strides
        tg.B.Output("y", tg.B.Scale(r, 2f));
        var s = tg.B.Slice(x, 2, 2, 4);                // [T,6,4], row stride 8
        tg.B.Output("z", tg.B.Add(tg.B.Reshape(s, 0, 24), tg.B.Reshape(s, 0, 24)));
        Diff.Check("Views.reshape", tg.B.Build(), new Step(3, 0, (x, Rnd.F(3 * 48, 1))), atol: 1e-6);
    }

    [Fact]
    public void PermuteHeadsThenMatMulAndBack()
    {
        Gpu.Require();
        int H = 4;
        int D = 16;
        var tg = new TG();
        var x = tg.InF("x", TG.T, H * D);
        var heads = tg.B.Reshape(x, 0, H, D);
        var w = tg.RandW("w", DataType.F32, 0.5f, 8, D);
        tg.B.Output("y", tg.B.MatMul(heads, w));
        tg.B.Output("sum", tg.B.Add(tg.B.Slice(x, 1, 0, 16), tg.B.Slice(x, 1, 16, 16)));
        Diff.Check("Views.heads", tg.B.Build(), new Step(7, 0, (x, Rnd.F(7 * H * D, 1))), atol: 2e-2, rtol: 1e-2);
    }

    [Fact]
    public void GraphOutputThatIsAView()
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", TG.T, 4, 10);
        tg.B.Output("v", tg.B.Permute(tg.B.Slice(x, 2, 2, 5), 0, 2, 1));
        Diff.Check("Views.output", tg.B.Build(), new Step(3, 0, (x, Rnd.F(3 * 40, 1))), atol: 0, rtol: 0);
    }
}
