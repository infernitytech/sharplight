using Sharplight.Graph;
using Sharplight.Runtime;

namespace Sharplight.Cpu;

// Matrix products and embedding lookups; weights stay in their (possibly quantized) storage format.
internal sealed unsafe partial class CpuCompiledSegment
{
    /// <summary>The value as a weight operand: weights/constants are used in place, activations must be dense.</summary>
    private TRef WeightOperand(Value v)
    {
        var r = Ref(v);
        return IsStatic(v) ? r : Dense(r);
    }

    private void LowerGather(Node n)
    {
        var ins = n.Inputs;
        var table = Ref(ins[0]);
        var indices = Ref(ins[1]);
        var y = NewDense(n.Output);
        var format = Quant.FormatOf(table.Type)!.Value;
        if (!IsStatic(ins[0]))
        {
            table = Dense(table);
        }

        AddStep(new GatherOp(table, indices, y, format), table, indices, y);
        _refs[n.Output] = y;
    }

    private void LowerMatMul(Node n)
    {
        var ins = n.Inputs;
        var x = RowContig(RefF32(ins[0]));
        var w = WeightOperand(ins[1]);
        var format = Quant.FormatOf(w.Type)!.Value;
        var y = NewDense(n.Output);
        bool intDot = _features.Contains(DeviceFeatures.IntDot) && (_options.AllowReducedPrecision || !n.Attributes.Get(StandardOps.F32ActivationsAttribute, false));
        var epilogue = AbsorbEpilogue(n.Output, y, columnBias: true);
        AddStep(new MatMulOp(x, w, y, format, intDot, IsStatic(ins[1]), epilogue), Touched(x, w, y, epilogue?.ColumnBias, epilogue?.Residual));
        _refs[n.Output] = y;
    }

    /// <summary>Mixture-of-experts matmul: the weight is [experts, N, K] and a per-row index selects the expert.</summary>
    private void LowerMatMulId(Node n)
    {
        var ins = n.Inputs;
        var x = RowContig(RefF32(ins[0]));
        var w = Ref(ins[1]);
        var expertIndex = Dense(Ref(ins[2]));
        if (!IsStatic(ins[1]))
        {
            w = Dense(w);
        }

        var format = Quant.FormatOf(w.Type)!.Value;
        var y = NewDense(n.Output);
        AddStep(new MatMulIdOp(x, w, expertIndex, y, format), x, w, expertIndex, y);
        _refs[n.Output] = y;
    }

    private void LowerBatchMatMul(Node n)
    {
        var a = n.Attributes;
        var x = RefF32(n.Inputs[0]);
        var z = RefF32(n.Inputs[1]);
        var y = NewDense(n.Output);
        AddStep(new BatchMatMulOp(x, z, y, a.Get("trans_a", false), a.Get("trans_b", false), a.Get("alpha", 1f)), x, z, y);
        _refs[n.Output] = y;
    }
}
