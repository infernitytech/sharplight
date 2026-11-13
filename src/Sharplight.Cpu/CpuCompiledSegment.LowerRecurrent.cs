using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Cpu;

// Recurrent and state-space ops that carry per-sequence state between runs.
internal sealed unsafe partial class CpuCompiledSegment
{
    private void LowerCausalConv1d(Node n)
    {
        var a = n.Attributes;
        var ins = n.Inputs;
        var x = RowContig(RefF32(ins[0]));
        var weight = RefF32(ins[1]);
        var state = Ref(ins[2]);
        var y = NewDense(n.Output);
        bool weightIsStatic = IsStatic(ins[1]);
        if (!weightIsStatic)
        {
            weight = Dense(weight);
        }

        // Optional bias (attribute "bias", or a non-i32 input 3) and, in the multi-sequence form, the trailing i32 sequence-runs table.
        bool hasBias = a.Get("bias", false) || (ins.Length > 3 && ins[3].Type != DataType.I32);
        TRef? bias = hasBias ? Dense(RefF32(ins[3])) : null;
        TRef? runs = ins.Length > (hasBias ? 4 : 3) ? Dense(Ref(ins[^1])) : null;
        var conv = new CausalConv1dOp(x, weight, state, y, a.Get("activation", "none") == "silu", weightIsStatic, bias, runs);
        AddStep(conv, Touched(x, weight, state, y, bias, runs));
        _refs[n.Output] = y;
    }

    private void LowerSelectiveScan(Node n)
    {
        var ins = n.Inputs;
        var x = RowContig(Ref(ins[0]));
        var dt = RowContig(Ref(ins[1]));
        var dtBias = Dense(RefF32(ins[2]));
        var a = Dense(RefF32(ins[3]));
        var b = RowContig(Ref(ins[4]));
        var c = RowContig(Ref(ins[5]));
        var d = Dense(RefF32(ins[6]));
        var state = Ref(ins[7]);
        var y = NewDense(n.Output);
        AddStep(new SelectiveScanOp(x, dt, dtBias, a, b, c, d, state, y, n.Attributes.Get("dt_min", 0f)), x, dt, dtBias, a, b, c, d, state, y);
        _refs[n.Output] = y;
    }

    private void LowerGatedDeltaRule(Node n)
    {
        var ins = n.Inputs;
        var q = RowContig(Ref(ins[0]));
        var k = RowContig(Ref(ins[1]));
        var v = RowContig(Ref(ins[2]));
        var g = Ref(ins[3]);
        var beta = Ref(ins[4]);
        var state = Ref(ins[5]);
        var y = NewDense(n.Output);
        TRef? runs = ins.Length > 6 ? Dense(Ref(ins[6])) : null;
        AddStep(new GatedDeltaRuleOp(q, k, v, g, beta, state, y, n.Attributes.Get("scale", 1f), runs), Touched(q, k, v, g, beta, state, runs, y));
        _refs[n.Output] = y;
    }
}
