namespace Sharplight.Cpu;

/// <summary>
/// One run's rollback plan (see <see cref="RollbackTables.Plan"/>): <see cref="Replay"/> record rows starting at
/// <see cref="ReadRecords"/> are applied before the run; a tentative run records its rows at <see cref="WriteRecords"/>
/// (null for a normal run) and does not store its advanced state.
/// </summary>
internal readonly unsafe struct RollbackRun(int replay, float* readRecords, float* writeRecords)
{
    /// <summary>Recorded rows applied to the state before the run.</summary>
    public int Replay { get; } = replay;

    /// <summary>First record row to replay.</summary>
    public float* ReadRecords { get; } = readRecords;

    /// <summary>First record row a tentative run writes; null for a normal run.</summary>
    public float* WriteRecords { get; } = writeRecords;

    /// <summary>The run's rows are tentative.</summary>
    public bool Tentative => WriteRecords != null;
}
