using Sharplight.Models;

namespace Sharplight.Graph;

/// <summary>
/// Rollback of recurrent state for speculative decoding: the contract every recurrent op (<c>CausalConv1d</c>,
/// <c>GatedDeltaRule</c>, <c>SelectiveScan</c>, <c>Wkv7</c>, <c>Wkv6</c>) implements in its rollback form, which
/// <see cref="MultiSequence.Transform(ComputeGraph, int, int, int, int)"/> emits when asked for rollback rows.
/// <para>
/// A speculative (tentative) run feeds a slot's pending token plus draft tokens, of which only a prefix will be kept. Instead of
/// snapshotting the whole state after every row (megabytes per layer and row for a delta-rule layer), the op leaves the slot's
/// state at the start of the run and records the run's per-row inputs (kilobytes); once the host knows how many rows were
/// accepted, the next run of the slot replays that many recorded rows into the state before its own rows, in the same kernel.
/// The extra memory is two banks of <see cref="Rows"/> record rows per slot and op; the extra bandwidth is writing the records.
/// </para>
/// Rollback form of a recurrent op (multi-sequence form plus two trailing inputs and attribute <see cref="RowsAttribute"/> = K):
/// <list type="bullet">
/// <item>input <c>commits</c> [U, 4] i32 (<see cref="InputRole.SequenceCommits"/>): per run (replay rows c, read bank b, tentative flag, 0);</item>
/// <item>input <c>records</c>, a state [S, 2, K, W] f32 (also written: declare it in the node's writes), W = <see cref="RecordWidth"/>.</item>
/// </list>
/// Per run u of slot s (first position p0, n rows): the state starts from zero when p0 = 0 (c is then ignored), otherwise from the
/// slot's state followed by the c rows recorded in bank b (records[s, b, 0..c)), applied exactly as the op applies rows (no
/// outputs). A tentative run (flag 1, n ≤ K) then stores that state back, records its own rows in bank 1 - b and computes its
/// outputs without storing the advanced state; a normal run stores the state after its last row. The layout of a record row is
/// defined by each provider (replay must reproduce the op's arithmetic bit for bit, so a provider may record what its kernel
/// actually reads); <see cref="RecordWidth"/> only fixes its size.
/// </summary>
public static class RecurrentRollback
{
    /// <summary>Attribute holding K, the most rows of one tentative run (and of one replay).</summary>
    public const string RowsAttribute = "rollback_rows";
    /// <summary>Name of the <see cref="InputRole.SequenceCommits"/> input of a graph in rollback form.</summary>
    public const string CommitsInput = "sequence_commits";
    /// <summary>Suffix appended to a recurrent state's name for its record state.</summary>
    public const string RecordStateSuffix = ".rollback";
    /// <summary>Record banks per slot (replay reads one while a tentative run writes the other).</summary>
    public const int Banks = 2;
    /// <summary>Columns of the commits table.</summary>
    public const int CommitColumns = 4;
    /// <summary>Inputs the rollback form appends to the multi-sequence form (commits, records).</summary>
    public const int TrailingInputs = 2;

    private static readonly HashSet<string> s_ops = ["CausalConv1d", "GatedDeltaRule", "SelectiveScan", "Wkv7", "Wkv6"];

    /// <summary>The recurrent ops that have a rollback form.</summary>
    public static IReadOnlySet<string> Ops => s_ops;

    /// <summary>Whether <paramref name="n"/> is a recurrent op in rollback form.</summary>
    public static bool IsRollbackForm(Node n) => n.Attributes.Has(RowsAttribute);

    /// <summary>K: most rows of one tentative run of <paramref name="n"/> (0 when the node is not in rollback form).</summary>
    public static int Rows(Node n) => (int)n.Attributes.Get(RowsAttribute, 0L);

    /// <summary>Number of inputs of <paramref name="n"/> without the rollback inputs (the multi-sequence form's inputs).</summary>
    public static int BaseInputCount(Node n) => IsRollbackForm(n) ? n.Inputs.Length - TrailingInputs : n.Inputs.Length;

    /// <summary>
    /// Floats per record row of a recurrent op, from its token inputs: <c>CausalConv1d</c> C (x [T, C]); <c>GatedDeltaRule</c>
    /// HV · (DK + DV + 2) (k, v, g and beta per value head); <c>SelectiveScan</c> H · (P + 1) + G · N (x and dt per head, B);
    /// <c>Wkv7</c> 5 · H · N (w, k, v, a, b); <c>Wkv6</c> 3 · H · N (k, v, w).
    /// </summary>
    /// <param name="op">Op name.</param>
    /// <param name="inputs">The op's inputs (single- or multi-sequence form).</param>
    /// <exception cref="NotSupportedException"><paramref name="op"/> has no rollback form.</exception>
    public static long RecordWidth(string op, IReadOnlyList<Value> inputs) => op switch
    {
        "CausalConv1d" => inputs[0].Shape[1].Fixed,
        "GatedDeltaRule" => inputs[2].Shape[1].Fixed * (inputs[0].Shape[2].Fixed + inputs[2].Shape[2].Fixed + 2),
        "SelectiveScan" => inputs[0].Shape[1].Fixed * (inputs[0].Shape[2].Fixed + 1) + inputs[4].Shape[1].Fixed * inputs[4].Shape[2].Fixed,
        "Wkv7" => 5 * inputs[0].Shape[1].Fixed * inputs[0].Shape[2].Fixed,
        "Wkv6" => 3 * inputs[0].Shape[1].Fixed * inputs[0].Shape[2].Fixed,
        _ => throw new NotSupportedException($"Op {op} has no rollback form."),
    };
}
