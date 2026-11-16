namespace Sharplight.Storage;

/// <summary>
/// The record of a move (<see cref="StorageMigration"/>), kept in the new configuration folder: the copied originals that are
/// deleted on the next start (<see cref="StorageMigration.CompletePending"/>), once the new location is committed.
/// </summary>
internal sealed record StorageMigrationJournal
{
    /// <summary>The new location was committed (marker or pointer written), so the originals may go.</summary>
    public bool Committed { get; init; }

    /// <summary>The configuration folder of the new layout; the journal applies only while that layout is in effect.</summary>
    public string DestinationConfig { get; init; } = "";

    /// <summary>The source folders, pruned of empty subfolders after the deletion.</summary>
    public IReadOnlyList<string> SourceRoots { get; init; } = [];

    /// <summary>The originals to delete.</summary>
    public IReadOnlyList<StorageMigrationJournalEntry> Files { get; init; } = [];
}
