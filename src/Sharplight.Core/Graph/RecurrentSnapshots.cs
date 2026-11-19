namespace Sharplight.Graph;

/// <summary>
/// In-step snapshots of recurrent state, for prefix caching of models with recurrent state: the contract of the snapshot form of the
/// recurrent ops (<c>CausalConv1d</c>, <c>GatedDeltaRule</c>, <c>SelectiveScan</c>, <c>Wkv7</c>, <c>Wkv6</c>), which
/// <see cref="MultiSequence.Transform(ComputeGraph, int, int, int, int, int)"/> emits for a graph with spare state rows when every op
/// with per-slot state has one.
/// <para>
/// A prefix-cache snapshot is the state after a block boundary. Without the snapshot form it can only be copied after a step, so a
/// boundary inside a prompt chunk costs that prompt one more pass (the chunk is cut there). In the snapshot form the op stores the state
/// at the boundary while it walks the run, in the same step.
/// </para>
/// Snapshot form (multi-sequence form plus attribute <see cref="Attribute"/> = true and one input right after the runs table, before
/// the rollback inputs): <c>snapshots</c> [U, 2] i32 (<see cref="Models.InputRole.SequenceSnapshots"/>): per run u (k, row). When
/// k &gt; 0 the op also stores the state as it is after the run's first k rows (after a rollback replay, which comes before the run's
/// rows) into state row <c>row</c> (a spare row, never a slot's), in the state's own format; 0 &lt; k ≤ the run's rows, and a run with a
/// snapshot is never tentative. The run's outputs and its slot's final state are those of the plain form (a provider may compute them
/// in a different order, e.g. a chunked kernel aligning a chunk end with row k, within its usual precision).
/// </summary>
public static class RecurrentSnapshots
{
    /// <summary>Attribute marking a recurrent op in snapshot form (bool).</summary>
    public const string Attribute = "state_snapshots";
    /// <summary>Name of the <see cref="Models.InputRole.SequenceSnapshots"/> input.</summary>
    public const string Input = "sequence_snapshots";
    /// <summary>Columns of the snapshots table.</summary>
    public const int Columns = 2;

    /// <summary>Whether <paramref name="n"/> is a recurrent op in snapshot form.</summary>
    public static bool IsSnapshotForm(Node n) => n.Attributes.Get(Attribute, false);

    /// <summary>Index of the snapshots table among <paramref name="n"/>'s inputs (right after the runs table), or -1 when not in snapshot form.</summary>
    public static int TableIndex(Node n) => IsSnapshotForm(n) ? RecurrentRollback.BaseInputCount(n) : -1;
}
