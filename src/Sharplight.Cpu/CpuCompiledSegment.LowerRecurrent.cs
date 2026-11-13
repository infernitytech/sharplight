using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Cpu;

// Recurrent and state-space ops that carry per-sequence state between runs.
internal sealed unsafe partial class CpuCompiledSegment
{
    private void LowerCausalConv1d(Node n)
    {
        var a = n.Attributes;
        var ins = BaseInputs(n);
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
        var (rollback, commits, records) = LowerRollback(n);
        var conv = new CausalConv1dOp(x, weight, state, y, a.Get("activation", "none") == "silu", weightIsStatic, bias, runs, rollback);
        AddStep(conv, Touched(x, weight, state, y, bias, runs, commits, records));
        _refs[n.Output] = y;
    }

    private void LowerSelectiveScan(Node n)
    {
        var ins = BaseInputs(n);
        var x = RowContig(Ref(ins[0]));
        var dt = RowContig(Ref(ins[1]));
        var dtBias = Dense(RefF32(ins[2]));
        var a = Dense(RefF32(ins[3]));
        var b = RowContig(Ref(ins[4]));
        var c = RowContig(Ref(ins[5]));
        var d = Dense(RefF32(ins[6]));
        var state = Ref(ins[7]);
        var y = NewDense(n.Output);
        TRef? runs = ins.Length > 8 ? Dense(Ref(ins[8])) : null;
        var (rollback, commits, records) = LowerRollback(n);
        AddStep(new SelectiveScanOp(x, dt, dtBias, a, b, c, d, state, y, n.Attributes.Get("dt_min", 0f), runs, rollback),
            Touched(x, dt, dtBias, a, b, c, d, state, runs, y, commits, records));
        _refs[n.Output] = y;
    }

    private void LowerGatedDeltaRule(Node n)
    {
        var ins = BaseInputs(n);
        var q = RowContig(Ref(ins[0]));
        var k = RowContig(Ref(ins[1]));
        var v = RowContig(Ref(ins[2]));
        var g = Ref(ins[3]);
        var beta = Ref(ins[4]);
        var state = Ref(ins[5]);
        var y = NewDense(n.Output);
        TRef? runs = ins.Length > 6 ? Dense(Ref(ins[6])) : null;
        var (rollback, commits, records) = LowerRollback(n);
        AddStep(new GatedDeltaRuleOp(q, k, v, g, beta, state, y, n.Attributes.Get("scale", 1f), runs, rollback),
            Touched(q, k, v, g, beta, state, runs, y, commits, records));
        _refs[n.Output] = y;
    }

    private void LowerWkv7(Node n)
    {
        var ins = BaseInputs(n);
        var r = RowContig(Ref(ins[0]));
        var w = RowContig(Ref(ins[1]));
        var k = RowContig(Ref(ins[2]));
        var v = RowContig(Ref(ins[3]));
        var a = RowContig(Ref(ins[4]));
        var b = RowContig(Ref(ins[5]));
        var state = Ref(ins[6]);
        var y = NewDense(n.Output);
        TRef? runs = ins.Length > 7 ? Dense(Ref(ins[7])) : null;
        var (rollback, commits, records) = LowerRollback(n);
        AddStep(new Wkv7Op(r, w, k, v, a, b, state, y, runs, rollback), Touched(r, w, k, v, a, b, state, runs, y, commits, records));
        _refs[n.Output] = y;
    }

    private void LowerWkv6(Node n)
    {
        var ins = BaseInputs(n);
        var r = RowContig(Ref(ins[0]));
        var k = RowContig(Ref(ins[1]));
        var v = RowContig(Ref(ins[2]));
        var w = RowContig(Ref(ins[3]));
        var u = Dense(RefF32(ins[4]));
        var state = Ref(ins[5]);
        var y = NewDense(n.Output);
        TRef? runs = ins.Length > 6 ? Dense(Ref(ins[6])) : null;
        var (rollback, commits, records) = LowerRollback(n);
        AddStep(new Wkv6Op(r, k, v, w, u, state, y, runs, rollback), Touched(r, k, v, w, u, state, runs, y, commits, records));
        _refs[n.Output] = y;
    }

    /// <summary>The inputs of a recurrent op without the trailing rollback inputs (see <see cref="RecurrentRollback"/>).</summary>
    private static Value[] BaseInputs(Node n) => [.. n.Inputs.Take(RecurrentRollback.BaseInputCount(n))];

    /// <summary>The rollback tables of a recurrent op in rollback form (commits table and record state), or nulls.</summary>
    private (RollbackTables? Tables, TRef? Commits, TRef? Records) LowerRollback(Node n)
    {
        if (!RecurrentRollback.IsRollbackForm(n))
        {
            return (null, null, null);
        }

        var commits = Dense(Ref(n.Inputs[^2]));
        var records = Ref(n.Inputs[^1]);
        return (new RollbackTables(commits, records), commits, records);
    }
}
