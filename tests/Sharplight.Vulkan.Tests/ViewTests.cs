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

    /// <summary>
    /// <c>After</c> passes its first input through unchanged once its second is computed: two independent chains, the second started after the
    /// first (as chunked work is sequenced), read through slices and permutes of the passed-through value.
    /// </summary>
    [Fact]
    public void AfterPassesTheValueThrough()
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", TG.T, 4, 6);
        var y = tg.InF("y", TG.T, 4, 6);
        var first = tg.B.Silu(tg.B.Scale(x, 2f));
        var second = tg.B.After(tg.B.Slice(y, 2, 1, 4), first);
        tg.B.Output("first", first);
        tg.B.Output("second", tg.B.Add(tg.B.Contiguous(tg.B.Permute(second, 0, 2, 1)), tg.B.Contiguous(tg.B.Permute(tg.B.Slice(first, 2, 0, 4), 0, 2, 1))));
        Diff.Check("Views.after", tg.B.Build(), new Step(5, 0, (x, Rnd.F(5 * 24, 1)), (y, Rnd.F(5 * 24, 2))), atol: 1e-6, rtol: 1e-6);
    }

    /// <summary>
    /// Concatenations of several inputs run as one launch: strided, permuted and dense inputs of different extents along an
    /// inner and the outer axis, f32 and integer elements.
    /// </summary>
    [Fact]
    public void ConcatManyInputsInOneLaunch()
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", TG.T, 4, 10);
        var y = tg.InF("y", TG.T, 3, 4);
        var i = tg.In("i", DataType.I32, TG.T, 5);
        var permuted = tg.B.Permute(tg.B.Slice(x, 2, 0, 3), 0, 2, 1);
        tg.B.Output("cat", tg.B.Concat(1, tg.B.Slice(x, 2, 6, 4), y, permuted, tg.B.Slice(x, 2, 1, 4), tg.B.Scale(y, 2f)));
        tg.B.Output("icat", tg.B.Concat(1, i, tg.B.Slice(i, 1, 1, 2), i));
        tg.B.Output("outer", tg.B.Concat(0, y, tg.B.Scale(y, -1f), y));
        var step = new Step(5, 0, (x, Rnd.F(5 * 40, 1)), (y, Rnd.F(5 * 12, 2)), (i, Rnd.I(5 * 5, 3, 1000)));
        Diff.Check("Views.concat-many", tg.B.Build(), step, atol: 0, rtol: 0);
    }

    /// <summary>
    /// Static concatenations whose leading axes have extent 1 compute their inputs in the output's slices: inputs that are
    /// read again after the concatenation (by elementwise ops and a matmul over a view), a view input and a slice that is
    /// not 256-byte aligned (both copied), two views copied by one launch next to a computed input, an f32 matmul over the
    /// concatenation, and the same inputs in the other order.
    /// </summary>
    [Fact]
    public void ConcatInputsComputedInPlace()
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", 1, 64, 32);
        var y = tg.InF("y", 1, 32, 32);
        var a = tg.InF("a", 3, 16);
        var w = tg.RandW("w", DataType.F32, 0.1f, 24, 32);
        var s = tg.B.Scale(x, 2f, 0.5f);
        var t = tg.B.Add(y, y);
        var cat = tg.B.Concat(1, s, t, tg.B.Slice(x, 1, 8, 16));
        tg.B.Output("mm", tg.B.MatMul(cat, w));
        tg.B.Output("sq", tg.B.Mul(s, s));
        tg.B.Output("smm", tg.B.MatMul(tg.B.Reshape(s, 64, 32), w));
        tg.B.Output("tcat", tg.B.Scale(tg.B.Concat(1, t, tg.B.Silu(tg.B.Slice(x, 1, 0, 8))), 3f));
        tg.B.Output("mixed", tg.B.Scale(tg.B.Concat(1, tg.B.Silu(y), tg.B.Slice(x, 1, 0, 8), tg.B.Slice(x, 1, 16, 8)), 1f, 1f));
        var ra = tg.B.Scale(a, -1f);
        tg.B.Output("rows", tg.B.Add(tg.B.Concat(0, ra, tg.B.Scale(a, 4f), ra), tg.B.Concat(0, a, a, a)));
        var step = new Step(1, 0, (x, Rnd.F(64 * 32, 1)), (y, Rnd.F(32 * 32, 2)), (a, Rnd.F(3 * 16, 3)));
        Diff.Check("Views.concat-in-place", tg.B.Build(), step, atol: 1e-4, rtol: 1e-4);
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
