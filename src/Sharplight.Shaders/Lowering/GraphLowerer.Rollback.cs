using Sharplight.Graph;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    /// <summary>
    /// Rollback form of a recurrent op (<see cref="RecurrentRollback"/>): the commits table and record state the kernel binds
    /// after the runs table, and the macros selecting the kernel's rollback path (<c>ROLLBACK</c>, <c>KR</c> record rows per
    /// bank, <c>RW</c> floats per record row). Empty for the plain forms, so their kernels are unchanged.
    /// </summary>
    private (TensorRef? Commits, TensorRef? Records, (string, object)[] Macros) RollbackBindings(Node n)
    {
        if (!RecurrentRollback.IsRollbackForm(n))
        {
            return (null, null, []);
        }

        var commits = DenseInput(n.Inputs[^2], "seq-commits");
        var records = Ref(n.Inputs[^1]);
        return (commits, records, [("ROLLBACK", true), ("KR", records.Shape[2]), ("RW", records.Shape[3])]);
    }

    /// <summary>
    /// The runs table of a multi-sequence recurrent op whose plain form has <paramref name="baseInputs"/> inputs (the last
    /// input before the rollback inputs), or null for the single-sequence form.
    /// </summary>
    private TensorRef? RecurrentRuns(Node n, int baseInputs)
    {
        int count = RecurrentRollback.BaseInputCount(n);
        return count > baseInputs ? DenseInput(n.Inputs[count - 1], "seq-runs") : null;
    }

    /// <summary>Tensors a recurrent launch reads besides its own inputs: runs, commits and records (each when present).</summary>
    private static TensorRef[] SequenceTables(TensorRef? runs, TensorRef? commits, TensorRef? records)
        => [.. new[] { runs, commits, records }.OfType<TensorRef>()];
}
