using Sharplight.Graph;

namespace Sharplight.Cpu;

/// <summary>
/// The rollback inputs of a recurrent op (see <see cref="RecurrentRollback"/>): the commits table [U, 4] and the record state
/// [S, 2, K, W]. Resolves per run how many recorded rows to replay, where they are and where a tentative run records its rows.
/// </summary>
internal sealed unsafe class RollbackTables
{
    private readonly TRef _commits;
    private readonly TRef _records;
    private readonly int _rows;
    private readonly long _width;

    public RollbackTables(TRef commits, TRef records)
    {
        _commits = commits;
        _records = records;
        _rows = (int)records.Dims[2];
        _width = records.Dims[3];
    }

    /// <summary>Floats per record row.</summary>
    public long Width => _width;

    /// <summary>
    /// The rollback plan of run <paramref name="u"/> of slot <paramref name="slot"/> with <paramref name="count"/> rows at first
    /// position <paramref name="position"/>: rows to replay (0 for a fresh sequence), the first replayed record row, the first
    /// record row a tentative run writes (null for a normal run).
    /// </summary>
    /// <exception cref="InvalidOperationException">The commits row is out of range (replay or tentative run longer than K).</exception>
    public RollbackRun Plan(long u, long slot, long count, long position)
    {
        int* c = (int*)_commits.P + u * RecurrentRollback.CommitColumns;
        int replay = position == 0 ? 0 : c[0];
        int bank = c[1];
        bool tentative = c[2] != 0;
        if (replay < 0 || replay > _rows || (uint)bank >= RecurrentRollback.Banks || (tentative && count > _rows))
        {
            throw new InvalidOperationException($"Bad rollback commit row {u}: replay {c[0]}, bank {bank}, tentative {c[2]} with {count} rows (K = {_rows}).");
        }

        float* read = Bank(slot, bank);
        float* write = tentative ? Bank(slot, 1 - bank) : null;
        return new RollbackRun(replay, read, write);
    }

    private float* Bank(long slot, int bank) => (float*)_records.P + (slot * RecurrentRollback.Banks + bank) * _rows * _width;
}
