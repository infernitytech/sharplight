using Sharplight.Graph;

namespace Sharplight.Cpu;

// Pooling variants (max-pool argmax indices, region-of-interest max pooling) and deformable convolution.
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

    private void LowerDeformConv(Node n)
    {
        var a = n.Attributes;
        var x = Dense(RefF32(n.Inputs[0]));
        var w = Dense(RefF32(n.Inputs[1]));
        var offset = Dense(RefF32(n.Inputs[2]));
        var bias = Dense(RefF32(n.Inputs[3]));
        var mask = n.Inputs.Length > 4 ? Dense(RefF32(n.Inputs[4])) : null;
        var y = NewDense(n.Output);
        var op = new DeformConvOp(x, w, offset, bias, mask, y, Ints(a, "strides", 1, 1), Ints(a, "pads", 0, 0, 0, 0), Ints(a, "dilations", 1, 1),
            (int)a.Get("groups", 1L), (int)a.Get("offset_groups", 1L));
        AddStep(op, mask is null ? [x, w, offset, bias, y] : [x, w, offset, bias, mask, y]);
        _refs[n.Output] = y;
    }
}
