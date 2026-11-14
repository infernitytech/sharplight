using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    /// <summary>Support check of the RWKV recurrences: all token inputs, <c>u</c> and the state are f32 (anything else is partitioned to the CPU provider).</summary>
    private static OpSupport SupportsWkv(Node node, int inputs)
    {
        for (int i = 0; i < inputs; i++)
        {
            if (node.Inputs[i].Type != DataType.F32)
            {
                return OpSupport.No($"{node.Op.Name} needs f32 inputs and state, got {node.Inputs[i].Type} for {node.Inputs[i].Name}");
            }
        }

        return OpSupport.Yes;
    }

    /// <summary>RWKV-7 recurrence: one workgroup per (head, block of state rows, run); the state is updated in place.</summary>
    private void LowerWkv7(Node n) => LowerWkv(n, "wkv7", stateInput: 6);

    /// <summary>RWKV-6 recurrence: same decomposition as <see cref="LowerWkv7"/>.</summary>
    private void LowerWkv6(Node n) => LowerWkv(n, "wkv6", stateInput: 5);

    private void LowerWkv(Node n, string entry, int stateInput)
    {
        // Token inputs first (dense [T, H, N]), then for Wkv6 the bonus u [H, N]; Wkv7 has two more token inputs (a, b) before the state.
        var tokenInputs = n.Inputs.Take(stateInput).Select((v, i) => EnsureDense(Ref(v), $"{entry}-{i}")).ToArray();
        var state = Ref(n.Inputs[stateInput]);
        var runs = RecurrentRuns(n, stateInput + 1);
        var (commits, records, rollback) = RollbackBindings(n);
        var y = Produce(n.Output);
        long H = tokenInputs[0].Shape[1];
        long N = tokenInputs[0].Shape[2];
        int lanesPerRow = _opt.Has(DeviceFeatures.SubgroupOps) ? Math.Min(DiagnosticSwitches.WkvLanesPerRow, _opt.SubgroupSize) : 1;
        while (N % lanesPerRow != 0 && lanesPerRow > 1)
        {
            lanesPerRow /= 2;
        }

        long rowsPerBlock = Math.Min(DiagnosticSwitches.WkvRowsPerBlock, N);
        while (rowsPerBlock > 1 && (N % rowsPerBlock != 0 || rowsPerBlock * lanesPerRow > 256))
        {
            rowsPerBlock /= 2;
        }

        long timeBlock = Math.Clamp(1024 / N, 1, 16);
        var k = Lib("wkv.slang", entry, Macros([("H", H), ("N", N), ("LPR", lanesPerRow), ("RB", rowsPerBlock), ("TB", timeBlock), ("MULTI_SEQ", runs is not null), .. rollback]));

        // Wkv7: r, w, k, v, a, b; Wkv6: r, k, v, w and u. The kernel binds (r, w, k, v, a-or-u, b-or-unused).
        TensorRef[] bound = entry == "wkv7"
            ? tokenInputs
            : [tokenInputs[0], tokenInputs[3], tokenInputs[1], tokenInputs[2], tokenInputs[4], tokenInputs[4]];
        Emit($"{entry} {n.Output.Name}", [n],
            [Always(k, b => new Grid((uint)H, (uint)(N / rowsPerBlock), runs is null ? 1u : (uint)Rows(b, runs)))],
            [.. bound.Select(r => Pointer(r)), Pointer(state), Pointer(y), .. SeqArgs(runs, commits, records), TokenArg(tokenInputs[0])],
            [.. tokenInputs, state, .. SequenceTables(runs, commits, records)], [y, state, .. records is null ? [] : (TensorRef[])[records]]);
    }
}
