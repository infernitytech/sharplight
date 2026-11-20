using Sharplight.Graph;

namespace Sharplight.Cpu;

/// <summary>
/// The snapshots table [U, 2] of a recurrent op in snapshot form (see <see cref="RecurrentSnapshots"/>): per run the rows after which the
/// op also stores the state, and the state row it stores it into.
/// </summary>
internal sealed unsafe class StateSnapshotTable(TRef table)
{
    /// <summary>
    /// Run <paramref name="u"/>'s snapshot: rows k (0: none) and the destination state row. Checked against the run's
    /// <paramref name="count"/> rows and the state's <paramref name="stateRows"/> rows.
    /// </summary>
    /// <exception cref="InvalidOperationException">The row is out of range.</exception>
    public (int Rows, long Row) At(long u, long count, long stateRows, string op)
    {
        int* p = (int*)table.P + u * RecurrentSnapshots.Columns;
        int k = p[0];
        if (k <= 0)
        {
            return (0, 0);
        }

        if (k > count || (uint)p[1] >= (ulong)stateRows)
        {
            throw new InvalidOperationException($"{op}: bad snapshot row {u}: {k} of {count} rows into state row {p[1]} of {stateRows}.");
        }

        return (k, p[1]);
    }
}
