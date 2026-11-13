using Sharplight.Graph;

namespace Sharplight.Cpu;

// Dense linear algebra beyond matrix products.
internal sealed unsafe partial class CpuCompiledSegment
{
    private void LowerDet(Node n)
    {
        var x = Dense(RefF32(n.Inputs[0]));
        var y = NewDense(n.Output);
        AddStep(new DetOp(x, y), x, y);
        _refs[n.Output] = y;
    }
}
