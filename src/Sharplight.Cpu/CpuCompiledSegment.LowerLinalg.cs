using Sharplight.Graph;

namespace Sharplight.Cpu;

// Dense linear algebra beyond matrix products and batched linear attention recurrences.
internal sealed unsafe partial class CpuCompiledSegment
{
    private void LowerDet(Node n)
    {
        var x = Dense(RefF32(n.Inputs[0]));
        var y = NewDense(n.Output);
        AddStep(new DetOp(x, y), x, y);
        _refs[n.Output] = y;
    }

    private void LowerInverse(Node n)
    {
        var x = Dense(RefF32(n.Inputs[0]));
        var y = NewDense(n.Output);
        AddStep(new InverseOp(x, y), x, y);
        _refs[n.Output] = y;
    }

    private void LowerLinearAttention(Node n)
    {
        var a = n.Attributes;
        var q = Dense(RefF32(n.Inputs[0]));
        var k = Dense(RefF32(n.Inputs[1]));
        var v = Dense(RefF32(n.Inputs[2]));
        int next = 3;
        TRef? Optional(string flag) => a.Get(flag, false) ? Dense(RefF32(n.Inputs[next++])) : null;
        var state = Optional("state");
        var decay = Optional("gated");
        var beta = Optional("delta");
        var y = NewDense(n.Outputs[0]);
        var stateOut = NewDense(n.Outputs[1]);
        var op = new LinearAttentionOp(q, k, v, state, decay, beta, y, stateOut, (int)a.Get<long>("q_heads"), (int)a.Get<long>("kv_heads"), a.Get("scale", 1f));
        AddStep(op, [q, k, v, .. new[] { state, decay, beta }.OfType<TRef>(), y, stateOut]);
        _refs[n.Outputs[0]] = y;
        _refs[n.Outputs[1]] = stateOut;
    }
}
