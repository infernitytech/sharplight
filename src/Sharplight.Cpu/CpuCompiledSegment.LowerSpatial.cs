using Sharplight.Graph;

namespace Sharplight.Cpu;

// Image-style ops on NCHW tensors, plus reductions and group normalization.
internal sealed unsafe partial class CpuCompiledSegment
{
    private void LowerConv(Node n)
    {
        var a = n.Attributes;
        var ins = n.Inputs;
        var x = Dense(RefF32(ins[0]));
        var weight = RefF32(ins[1]);
        TRef? bias = ins.Length > 2 ? Dense(RefF32(ins[2])) : null;
        var y = NewDense(n.Output);
        var conv = ConvOp.Create(
            x, (float*)(weight.Buf.Ptr + weight.Offset * 4), (int)weight.Dims[2], (int)weight.Dims[3], bias, y, (int)a.Get("groups", 1L),
            Ints(a, "strides", 1, 1), Ints(a, "pads", 0, 0, 0, 0), Ints(a, "dilations", 1, 1));
        AddStep(conv, Touched(x, bias, y));
        _refs[n.Output] = y;
    }

    private void LowerConvTranspose(Node n)
    {
        var a = n.Attributes;
        var ins = n.Inputs;
        var x = Dense(RefF32(ins[0]));
        var weight = RefF32(ins[1]);
        TRef? bias = ins.Length > 2 ? Dense(RefF32(ins[2])) : null;
        var y = NewDense(n.Output);
        var strides = Ints(a, "strides", 1, 1);
        var pads = Ints(a, "pads", 0, 0, 0, 0);
        var dilations = Ints(a, "dilations", 1, 1);
        var convTranspose = new ConvTransposeOp(
            x, (float*)(weight.Buf.Ptr + weight.Offset * 4), (int)weight.Dims[2], (int)weight.Dims[3], bias, y, (int)a.Get("groups", 1L),
            strides[0], strides[1], pads[0], pads[1], dilations[0], dilations[1]);
        AddStep(convTranspose, Touched(x, bias, y));
        _refs[n.Output] = y;
    }

    private void LowerPool(Node n)
    {
        var a = n.Attributes;
        var x = Dense(RefF32(n.Inputs[0]));
        var y = NewDense(n.Output);
        var pool = new PoolOp(
            x, y, a.Get("mode", "max") == "max", Ints(a, "kernel"), Ints(a, "strides", 1, 1), Ints(a, "pads", 0, 0, 0, 0),
            Ints(a, "dilations", 1, 1), a.Get("count_include_pad", false));
        AddStep(pool, x, y);
        _refs[n.Output] = y;
    }

    private void LowerReduce(Node n)
    {
        var a = n.Attributes;
        var x = RefF32(n.Inputs[0]);
        var y = NewDense(n.Output);
        var reduced = new bool[x.Rank];
        foreach (var axis in a.Get<long[]>("axes"))
        {
            reduced[axis < 0 ? axis + x.Rank : axis] = true;
        }

        var kind = a.Get("mode", "sum") switch
        {
            "mean" => ReduceKind.Mean,
            "max" => ReduceKind.Max,
            "min" => ReduceKind.Min,
            "prod" => ReduceKind.Prod,
            "l1" => ReduceKind.L1,
            "l2" => ReduceKind.L2,
            "sumsquare" => ReduceKind.SumSquare,
            "logsumexp" => ReduceKind.LogSumExp,
            _ => ReduceKind.Sum,
        };
        AddStep(new ReduceOp(kind, x, y, reduced), x, y);
        _refs[n.Output] = y;
    }

    private void LowerPad(Node n)
    {
        var a = n.Attributes;
        var x = RefF32(n.Inputs[0]);
        var y = NewDense(n.Output);
        AddStep(new PadOp(x, y, a.Get<long[]>("pads"), a.Get("mode", "constant"), a.Get("value", 0f)), x, y);
        _refs[n.Output] = y;
    }

    private void LowerResize(Node n)
    {
        var a = n.Attributes;
        var x = Dense(RefF32(n.Inputs[0]));
        var y = NewDense(n.Output);
        AddStep(new ResizeOp(x, y, a.Get("mode", "nearest"), a.Get("coord", "half_pixel"), a.Get("nearest_mode", "round_prefer_floor")), x, y);
        _refs[n.Output] = y;
    }

    private void LowerGridSample(Node n)
    {
        var a = n.Attributes;
        var x = Dense(RefF32(n.Inputs[0]));
        var grid = Dense(RefF32(n.Inputs[1]));
        var y = NewDense(n.Output);
        AddStep(new GridSampleOp(x, grid, y, a.Get("mode", "linear"), a.Get("padding", "zeros"), a.Get("align_corners", false)), x, grid, y);
        _refs[n.Output] = y;
    }

    private void LowerRoiAlign(Node n)
    {
        var a = n.Attributes;
        var x = Dense(RefF32(n.Inputs[0]));
        var rois = Dense(RefF32(n.Inputs[1]));
        var batch = Dense(Ref(n.Inputs[2]));
        var y = NewDense(n.Output);
        var op = new RoiAlignOp(x, rois, batch, y, (int)a.Get("sampling_ratio", 0L), a.Get("spatial_scale", 1f), a.Get("mode", "avg") == "max", a.Get("half_pixel", true));
        AddStep(op, x, rois, batch, y);
        _refs[n.Output] = y;
    }

    private void LowerGroupNorm(Node n)
    {
        var a = n.Attributes;
        var ins = n.Inputs;
        var x = Dense(RefF32(ins[0]));
        TRef? weight = ins.Length > 1 ? Dense(RefF32(ins[1])) : null;
        TRef? bias = ins.Length > 2 ? Dense(RefF32(ins[2])) : null;
        var y = NewDense(n.Output);
        AddStep(new GroupNormOp(x, weight, bias, y, (int)a.Get<long>("groups"), a.Get("eps", 1e-5f)), Touched(x, y, weight, bias));
        _refs[n.Output] = y;
    }
}
