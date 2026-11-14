using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Cpu;

// Integer elementwise arithmetic and Narrow (StandardOps.Numeric).
internal sealed unsafe partial class CpuCompiledSegment
{
    private void LowerInteger(Node n)
    {
        var a = Ref(n.Inputs[0]);
        TRef? b = n.Inputs.Length > 1 ? Ref(n.Inputs[1]) : null;
        var y = NewDense(n.Output);
        var kind = Enum.Parse<IntOpKind>(n.Op.Name);
        var op = new IntElementwiseOp(kind, a, b, y, n.Attributes.Get("fmod", false), n.Attributes.Get("unsigned", false));
        AddStep(op, b is null ? [a, y] : [a, b, y]);
        _refs[n.Output] = y;
    }

    private void LowerNarrow(Node n)
    {
        var a = n.Attributes;
        var x = RowContig(Ref(n.Inputs[0]));
        var y = NewDense(n.Output);
        var format = NumericFormats.Parse(a.Get<string>("format"));
        var rounding = Enum.Parse<ExponentRounding>(a.Get("rounding", "up"), ignoreCase: true);
        AddStep(new NarrowOp(x, y, format, a.Get("saturate", true), rounding), x, y);
        _refs[n.Output] = y;
    }
}
