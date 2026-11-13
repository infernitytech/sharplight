using Sharplight.Graph;

namespace Sharplight.Cpu;

// Pooling variants: max-pool argmax indices and region-of-interest max pooling.
internal sealed unsafe partial class CpuCompiledSegment
{
    private void LowerPoolArgMax(Node n)
    {
        var a = n.Attributes;
        var x = Dense(RefF32(n.Inputs[0]));
        var y = NewDense(n.Output);
        var op = new PoolArgMaxOp(x, y, Ints(a, "kernel"), Ints(a, "strides", 1, 1), Ints(a, "pads", 0, 0, 0, 0), Ints(a, "dilations", 1, 1),
            a.Get("storage_order", 0L) == 1);
        AddStep(op, x, y);
        _refs[n.Output] = y;
    }

    private void LowerMaxRoiPool(Node n)
    {
        var x = Dense(RefF32(n.Inputs[0]));
        var rois = Dense(RefF32(n.Inputs[1]));
        var y = NewDense(n.Output);
        AddStep(new MaxRoiPoolOp(x, rois, y, n.Attributes.Get("spatial_scale", 1f)), x, rois, y);
        _refs[n.Output] = y;
    }
}
