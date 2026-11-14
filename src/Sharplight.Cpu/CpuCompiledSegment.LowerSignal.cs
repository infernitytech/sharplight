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
}
