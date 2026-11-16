using System.Security.Cryptography;
using System.Text.Json;

namespace Sharplight.Storage;

/// <summary>
/// Changes the storage location (<see cref="StorageLayout"/>) and takes the existing files along: copies every file of the chosen
/// categories into the new layout and verifies each copy (SHA-256 of the original while reading it, compared with a second read of
/// the copy, before the copy gets its final name), then records the new mode (<see cref="StorageSelection"/>). A move deletes the
/// originals only afterwards, on the next start (<see cref="CompletePending"/>), and only those still unchanged since they were copied:
/// an interruption at any point leaves the old location complete and in effect, or the new one complete and committed. The model
/// store of a move is renamed in one step when both folders are on the same volume.
/// </summary>
public sealed class StorageMigration
{
    /// <summary>File name of the move journal in the new configuration folder.</summary>
    public const string JournalFileName = "storage-migration.json";

    private const string PartialSuffix = ".sl-partial";
    private const int BufferSize = 1 << 20;
    private static readonly string[] s_skippedRootFiles = [StoragePointer.FileName, JournalFileName];

    /// <summary>Prepares a change from <paramref name="source"/> to <paramref name="destination"/>.</summary>
    /// <param name="source">The layout in effect (<see cref="StorageLayout.Resolve"/>).</param>
    /// <param name="destination">The new layout (<see cref="StorageLayout.Create"/>).</param>
    /// <param name="categories">The kinds of files to take along (e.g. everything but <see cref="StorageCategory.Models"/>).</param>
    public StorageMigration(StorageLayout source, StorageLayout destination, IEnumerable<StorageCategory> categories)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(destination);
        ArgumentNullException.ThrowIfNull(categories);
        Source = source;
        Destination = destination;
        var comparison = PackageLocation.Comparison(source.Platform);
        Items = [.. categories.Distinct().Order()
            .Select(category => new StorageMigrationItem(category, PackageLocation.Normalize(source.GetDirectory(category)),
                PackageLocation.Normalize(destination.GetDirectory(category))))
            .Where(item => !string.Equals(item.Source, item.Destination, comparison))];
    }

    /// <summary>The layout in effect.</summary>
    public StorageLayout Source { get; }

    /// <summary>The new layout.</summary>
    public StorageLayout Destination { get; }

    /// <summary>The folders to transfer (categories whose folder does not change are left out).</summary>
    public IReadOnlyList<StorageMigrationItem> Items { get; }

    /// <summary>
    /// Why the change cannot be made, or null: a new folder inside a current one (or the other way round) would copy into itself.
    /// </summary>
    public string? Validate()
    {
        var comparison = PackageLocation.Comparison(Source.Platform);
        foreach (var item in Items)
        {
            foreach (var other in Items)
            {
                if (PackageLocation.IsInside(other.Destination, item.Source, comparison))
                {
                    return $"The new folder {other.Destination} is inside the current folder {item.Source}.";
                }

                if (PackageLocation.IsInside(item.Source, other.Destination, comparison))
                {
                    return $"The current folder {item.Source} is inside the new folder {other.Destination}.";
                }
            }
        }

        return null;
    }

    /// <summary>
    /// Transfers the files (<paramref name="kind"/>) and records the new mode. After a move the originals are deleted by
    /// <see cref="CompletePending"/> at the next start, once nothing has them open.
    /// </summary>
    /// <param name="kind">Move, copy or start fresh.</param>
    /// <param name="progress">Receives progress, or <see langword="null"/>.</param>
    /// <param name="cancellationToken">
    /// Stops between buffers while copying; the old location stays in effect (copies made so far are reused by the next attempt).
    /// </param>
    /// <returns>What was transferred.</returns>
    /// <exception cref="InvalidOperationException">The change cannot be made (<see cref="Validate"/>).</exception>
    public async Task<StorageMigrationResult> RunAsync(StorageTransferKind kind, IProgress<StorageMigrationProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        // A journal left by an earlier change into the same location that was given up before the restart must not delete anything.
        var journalFile = Path.Combine(Destination.ConfigDirectory, JournalFileName);
        StorageFiles.DeleteIfExists(journalFile);
        var journal = new StorageMigrationJournal { DestinationConfig = PackageLocation.Normalize(Destination.ConfigDirectory) };
        var result = new StorageMigrationResult(0, 0, []);
        if (kind != StorageTransferKind.StartFresh)
        {
            if (Validate() is { } problem)
            {
                throw new InvalidOperationException(problem);
            }

            var copied = new List<StorageMigrationJournalEntry>();
            result = await CopyAsync(kind == StorageTransferKind.Move, copied, progress, cancellationToken).ConfigureAwait(false);
            journal = journal with { SourceRoots = [.. Items.Select(item => item.Source)], Files = copied };
        }

        if (kind == StorageTransferKind.Move)
        {
            WriteJournal(journalFile, journal);
        }

        try
        {
            StorageSelection.Apply(Source, Destination.Mode, Destination.Root);
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            UndoRenames(result);
            throw;
        }

        if (kind == StorageTransferKind.Move)
        {
            WriteJournal(journalFile, journal with { Committed = true });
        }

        return result;
    }

    /// <summary>
    /// Finishes a move into <paramref name="layout"/> (the layout in effect, at start-up before anything opens its files): deletes the
    /// originals listed in its journal that are unchanged since they were copied and whose copy is still there, prunes the emptied
    /// folders and removes the journal. A journal of a move that was never committed is removed without deleting anything.
    /// </summary>
    /// <param name="layout">The layout in effect.</param>
    /// <returns>The number of originals deleted.</returns>
    public static int CompletePending(StorageLayout layout)
    {
        ArgumentNullException.ThrowIfNull(layout);
        var path = Path.Combine(layout.ConfigDirectory, JournalFileName);
        var journal = ReadJournal(path);
        if (journal is null)
        {
            return 0;
        }

        var comparison = PackageLocation.Comparison(layout.Platform);
        if (!string.Equals(journal.DestinationConfig, PackageLocation.Normalize(layout.ConfigDirectory), comparison))
        {
            return 0;
        }

        int deleted = 0;
        if (journal.Committed)
        {
            foreach (var entry in journal.Files)
            {
                deleted += TryDeleteOriginal(entry) ? 1 : 0;
            }

            foreach (var root in journal.SourceRoots)
            {
                PruneEmptyDirectories(root);
            }
        }

        StorageFiles.DeleteIfExists(path);
        return deleted;
    }

    private async Task<StorageMigrationResult> CopyAsync(bool move, List<StorageMigrationJournalEntry> copied,
        IProgress<StorageMigrationProgress>? progress, CancellationToken cancellationToken)
    {
        // The model store of a move is renamed last, right before the commit, so a cancelled or failed copy leaves it in place.
        var deferred = move ? Items.FirstOrDefault(item => item.Category == StorageCategory.Models) : null;
        var files = Items.Where(item => item != deferred).SelectMany(item => Enumerate(item).Select(file => (Item: item, File: file))).ToList();
        var deferredFiles = deferred is null ? [] : Enumerate(deferred).Select(file => (Item: deferred, File: file)).ToList();
        var counter = new CopyCounter(files.Sum(entry => entry.File.Length) + deferredFiles.Sum(entry => entry.File.Length), files.Count + deferredFiles.Count);
        await CopyFilesAsync(files, counter, copied, progress, cancellationToken).ConfigureAwait(false);
        cancellationToken.ThrowIfCancellationRequested();

        List<StorageCategory> renamed = [];
        if (deferred is not null && TryRename(deferred.Source, deferred.Destination))
        {
            renamed.Add(deferred.Category);
            counter.Add(deferredFiles.Sum(entry => entry.File.Length), deferredFiles.Count);
            progress?.Report(counter.Snapshot(null));
        }
        else
        {
            await CopyFilesAsync(deferredFiles, counter, copied, progress, cancellationToken).ConfigureAwait(false);
        }

        return new StorageMigrationResult(counter.Files, counter.Bytes, renamed);
    }

    private static async Task CopyFilesAsync(List<(StorageMigrationItem Item, FileInfo File)> files, CopyCounter counter,
        List<StorageMigrationJournalEntry> copied, IProgress<StorageMigrationProgress>? progress, CancellationToken cancellationToken)
    {
        foreach (var (item, file) in files)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var relative = Path.GetRelativePath(item.Source, file.FullName);
            var target = Path.Combine(item.Destination, relative);
            var reporter = progress is null ? null : new SynchronousProgress<long>(bytes => progress.Report(counter.Snapshot(relative, bytes)));
            await CopyVerifiedAsync(file, target, reporter, cancellationToken).ConfigureAwait(false);
            copied.Add(new StorageMigrationJournalEntry(file.FullName, target, file.Length, file.LastWriteTimeUtc.Ticks));
            counter.Add(file.Length, 1);
            progress?.Report(counter.Snapshot(null));
        }
    }

    /// <summary>The files of <paramref name="item"/>, without the folders of other categories nested in it and without stale partial copies.</summary>
    private IEnumerable<FileInfo> Enumerate(StorageMigrationItem item)
    {
        if (!Directory.Exists(item.Source))
        {
            return [];
        }

        var comparison = PackageLocation.Comparison(Source.Platform);
        var options = new EnumerationOptions { RecurseSubdirectories = true, IgnoreInaccessible = false, AttributesToSkip = FileAttributes.ReparsePoint };
        return new DirectoryInfo(item.Source).EnumerateFiles("*", options)
            .Where(file => !file.Name.EndsWith(PartialSuffix, StringComparison.Ordinal) && !file.Name.StartsWith(".write-probe-", StringComparison.Ordinal))
            .Where(file => !(item.Category == StorageCategory.Config && s_skippedRootFiles.Contains(file.Name)
                && string.Equals(file.DirectoryName, item.Source, comparison)))
            .Where(file => !IsExcluded(file.FullName, item, comparison))
            .ToList();
    }

    /// <summary>True when <paramref name="path"/> belongs to another category whose folder is nested in <paramref name="item"/>'s folder.</summary>
    private bool IsExcluded(string path, StorageMigrationItem item, StringComparison comparison)
    {
        foreach (var category in Enum.GetValues<StorageCategory>())
        {
            var folder = PackageLocation.Normalize(Source.GetDirectory(category));
            if (category != item.Category && !string.Equals(folder, item.Source, comparison) && PackageLocation.IsInside(folder, item.Source, comparison)
                && PackageLocation.IsInside(path, folder, comparison))
            {
                return true;
            }
        }

        return false;
    }

    private void UndoRenames(StorageMigrationResult result)
    {
        foreach (var item in Items.Where(item => result.RenamedCategories.Contains(item.Category)))
        {
            TryRename(item.Destination, item.Source);
        }
    }

    private static bool TryRename(string source, string destination)
    {
        if (!Directory.Exists(source))
        {
            return false;
        }

        try
        {
            if (Directory.Exists(destination))
            {
                if (Directory.EnumerateFileSystemEntries(destination).Any())
                {
                    return false;
                }

                Directory.Delete(destination);
            }

            Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
            Directory.Move(source, destination);
            return true;
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            // Another volume, or a file in use (Windows): copied instead.
            return false;
        }
    }

    private static async Task CopyVerifiedAsync(FileInfo source, string target, IProgress<long>? progress, CancellationToken cancellationToken)
    {
        var existing = new FileInfo(target);
        if (existing.Exists && existing.Length == source.Length && existing.LastWriteTimeUtc == source.LastWriteTimeUtc)
        {
            return;   // copied by an earlier, interrupted run
        }

        Directory.CreateDirectory(Path.GetDirectoryName(target)!);
        var partial = target + PartialSuffix;
        byte[] expected;
        await using (var input = new FileStream(source.FullName, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete, BufferSize, useAsync: true))
        await using (var output = new FileStream(partial, FileMode.Create, FileAccess.Write, FileShare.None, BufferSize, useAsync: true))
        {
            using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            var buffer = new byte[BufferSize];
            long written = 0;
            int read;
            while ((read = await input.ReadAsync(buffer, cancellationToken).ConfigureAwait(false)) > 0)
            {
                hash.AppendData(buffer, 0, read);
                await output.WriteAsync(buffer.AsMemory(0, read), cancellationToken).ConfigureAwait(false);
                written += read;
                progress?.Report(written);
            }

            await output.FlushAsync(cancellationToken).ConfigureAwait(false);
            output.Flush(flushToDisk: true);
            expected = hash.GetHashAndReset();
        }

        byte[] actual;
        await using (var check = new FileStream(partial, FileMode.Open, FileAccess.Read, FileShare.Read, BufferSize, useAsync: true))
        {
            actual = await SHA256.HashDataAsync(check, cancellationToken).ConfigureAwait(false);
        }

        if (!actual.AsSpan().SequenceEqual(expected))
        {
            File.Delete(partial);
            throw new IOException($"The copy of {source.FullName} does not match the original.");
        }

        File.SetLastWriteTimeUtc(partial, source.LastWriteTimeUtc);
        File.Move(partial, target, overwrite: true);
    }

    private static bool TryDeleteOriginal(StorageMigrationJournalEntry entry)
    {
        try
        {
            var original = new FileInfo(entry.Path);
            var copy = new FileInfo(entry.Destination);
            if (!original.Exists || !copy.Exists || copy.Length != entry.Length || original.Length != entry.Length
                || original.LastWriteTimeUtc.Ticks != entry.LastWriteTicks)
            {
                return false;   // changed after it was copied, or the copy is gone: kept
            }

            original.Delete();
            return true;
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            return false;
        }
    }

    private static void PruneEmptyDirectories(string root)
    {
        if (!Directory.Exists(root))
        {
            return;
        }

        try
        {
            foreach (var directory in Directory.EnumerateDirectories(root, "*", SearchOption.AllDirectories).OrderByDescending(path => path.Length).ToList())
            {
                if (!Directory.EnumerateFileSystemEntries(directory).Any())
                {
                    Directory.Delete(directory);
                }
            }

            if (!Directory.EnumerateFileSystemEntries(root).Any())
            {
                Directory.Delete(root);
            }
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            // Left for the user; the files that mattered are gone or kept on purpose.
        }
    }

    private static void WriteJournal(string path, StorageMigrationJournal journal)
    {
        StorageFiles.WriteAtomically(path, JsonSerializer.Serialize(journal, StorageJsonContext.Default.StorageMigrationJournal));
    }

    private static StorageMigrationJournal? ReadJournal(string path)
    {
        try
        {
            return File.Exists(path) ? JsonSerializer.Deserialize(File.ReadAllText(path), StorageJsonContext.Default.StorageMigrationJournal) : null;
        }
        catch (Exception error) when (error is JsonException or IOException or UnauthorizedAccessException)
        {
            return null;
        }
    }
}
