using Sharplight.Graph;

namespace Sharplight.Cpu;

// Elementwise, row-wise and select ops.
internal sealed unsafe partial class CpuCompiledSegment
{
    private void LowerUnary(Node n, UnaryKind kind)
    {
        if (CpuOpSupport.IsIntegerElementwise(n))
        {
            LowerInteger(n);
            return;
        }

        var x = RowContig(RefF32(n.Inputs[0]));
        var y = NewDense(n.Output);
        var (scale, bias) = UnaryParameters(kind, n.Attributes);
        AddStep(new UnaryOp(kind, x, y, scale, bias), x, y);
        _refs[n.Output] = y;
    }

    /// <summary>The two <see cref="UnaryOp"/> parameters of a unary node of <paramref name="kind"/> (Clip: min and max).</summary>
    private static (float Scale, float Bias) UnaryParameters(UnaryKind kind, Attributes a) => kind switch
    {
        UnaryKind.HardSigmoid => (a.Get("alpha", 0.2f), a.Get("beta", 0.5f)),
        UnaryKind.Elu or UnaryKind.Celu or UnaryKind.ThresholdedRelu => (a.Get("alpha", 1f), 0f),
        UnaryKind.Selu => (a.Get("alpha", 1.67326319217681884765625f), a.Get("gamma", 1.05070102214813232421875f)),
        UnaryKind.LeakyRelu => (a.Get("alpha", 0.01f), 0f),
        UnaryKind.Clip => (a.Get("min", float.NegativeInfinity), a.Get("max", float.PositiveInfinity)),
        _ => (1f, 0f),
    };

    private void LowerClip(Node n) => LowerUnary(n, UnaryKind.Clip);

    private void LowerScale(Node n)
    {
        var a = n.Attributes;
        var x = RowContig(RefF32(n.Inputs[0]));
        var y = NewDense(n.Output);
        AddStep(new UnaryOp(UnaryKind.Scale, x, y, a.Get("scale", 1f), a.Get("bias", 0f)), x, y);
        _refs[n.Output] = y;
    }

    private void LowerBinary(Node n, BinaryKind kind)
    {
        if (CpuOpSupport.IsIntegerElementwise(n))
        {
            LowerInteger(n);
            return;
        }

        var x = RefF32(n.Inputs[0]);
        var z = RefF32(n.Inputs[1]);
        var y = NewDense(n.Output);
        AddStep(new BinaryOp(kind, x, z, y, n.Attributes.Get("fmod", false)), x, z, y);
        _refs[n.Output] = y;
    }

    /// <summary>Row-wise ops (norms, softmax, row reductions) with optional weight and bias inputs.</summary>
    private void LowerRow(Node n, RowKind kind)
    {
        var a = n.Attributes;
        var ins = n.Inputs;
        var x = RowContig(RefF32(ins[0]));
        TRef? weight = ins.Length > 1 ? RefF32(ins[1]) : null;
        TRef? bias = ins.Length > 2 ? RefF32(ins[2]) : null;
        var y = NewDense(n.Output);
        AddStep(new RowOp(kind, x, weight, bias, y, a.Get("eps", 1e-5f), a.Get("scale", 1f), a.Get("log", false)), Touched(x, y, weight, bias));
        _refs[n.Output] = y;
    }

    private void LowerWhere(Node n)
    {
        var ins = n.Inputs;
        var condition = Ref(ins[0]);
        var x = Ref(ins[1]);
        var z = Ref(ins[2]);
        var y = NewDense(n.Output);
        AddStep(new WhereOp(condition, x, z, y), condition, x, z, y);
        _refs[n.Output] = y;
    }
}
