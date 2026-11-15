using Sharplight.Graph;

namespace Sharplight.Cpu;

// Signal processing.
internal sealed unsafe partial class CpuCompiledSegment
{
    private void LowerDft(Node n)
    {
        var a = n.Attributes;
        var x = Dense(RefF32(n.Inputs[0]));
        var y = NewDense(n.Output);
        AddStep(new DftOp(x, y, (int)a.Get<long>("length"), (int)a.Get<long>("out_bins"), a.Get("inverse", false), a.Get("real_output", false)), x, y);
        _refs[n.Output] = y;
    }

    private void LowerConv1d(Node n)
    {
        var a = n.Attributes;
        var x = Dense(RefF32(n.Inputs[0]));
        var w = Dense(RefF32(n.Inputs[1]));
        TRef? bias = n.Inputs.Length > 2 ? Dense(RefF32(n.Inputs[2])) : null;
        var y = NewDense(n.Output);
        AddStep(new Conv1dOp(x, w, bias, y, (int)a.Get("stride", 1L), (int)a.Get("pad_left", 0L), (int)a.Get("dilation", 1L), (int)a.Get("groups", 1L)),
            Touched(x, y, w, bias));
        _refs[n.Output] = y;
    }
}
