namespace Sharplight.Storage;

/// <summary>
/// An original of a <see cref="StorageMigrationJournal"/>: deleted only while it is unchanged since it was copied and its copy is
/// still there.
/// </summary>
/// <param name="Path">The original file.</param>
/// <param name="Destination">Its verified copy.</param>
/// <param name="Length">Its length when it was copied.</param>
/// <param name="LastWriteTicks">Its last write time (UTC ticks) when it was copied.</param>
internal sealed record StorageMigrationJournalEntry(string Path, string Destination, long Length, long LastWriteTicks);
