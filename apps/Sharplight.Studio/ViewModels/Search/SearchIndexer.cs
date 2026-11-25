using System.Security.Cryptography;
using System.Diagnostics;
using Sharplight.Llm.Embeddings;
using Sharplight.Studio.Services.Search;

namespace Sharplight.Studio.ViewModels.Search;

/// <summary>
/// Builds and updates the index of a folder. Files whose size and modification time match the previous index are kept; a file with another
/// time is hashed and re-embedded only when its content changed; new files are added and files that are gone are dropped. Progress is
/// checkpointed, so a cancelled or crashed run resumes where it stopped: the next run finds the finished files unchanged.
/// </summary>
/// <param name="parser">Reads the documents.</param>
internal sealed class SearchIndexer(IDocumentParser parser)
{
    /// <summary>The most documents one folder indexes.</summary>
    public const int MaxFiles = 5000;

    /// <summary>The most passages one document contributes.</summary>
    public const int MaxChunksPerFile = 4000;

    private const int BatchSize = 16;
    private const int CheckpointFiles = 25;
    private static readonly TimeSpan s_checkpointInterval = TimeSpan.FromSeconds(30);
    private static readonly HashSet<string> s_skippedFolders = new(StringComparer.OrdinalIgnoreCase) { ".git", ".hg", ".svn", "node_modules" };

    /// <summary>Brings the index of <paramref name="folder"/> up to date.</summary>
    /// <param name="folder">The folder.</param>
    /// <param name="previous">The stored index of the folder, or null. It is reused only for the same model and passage settings.</param>
    /// <param name="modelId">The embedding model's id.</param>
    /// <param name="embedder">The embedding model.</param>
    /// <param name="documentPrompt">The embedder's prompt for documents, or null.</param>
    /// <param name="chunkCharacters">The passage length.</param>
    /// <param name="overlapCharacters">The passage overlap.</param>
    /// <param name="report">Progress (fraction or null, text).</param>
    /// <param name="checkpoint">Called with the index so far every few files and when the run is cancelled; may store it.</param>
    /// <param name="cancellationToken">Cancels the run (after a last checkpoint).</param>
    public async Task<IndexUpdate> UpdateAsync(string folder, SearchIndex? previous, string modelId, ITextEmbedder embedder, string? documentPrompt,
        int chunkCharacters, int overlapCharacters, Action<double?, string> report, Action<SearchIndex>? checkpoint, CancellationToken cancellationToken)
    {
        folder = Path.TrimEndingDirectorySeparator(Path.GetFullPath(folder));
        bool reusable = previous is not null && previous.ModelId == modelId && previous.ChunkCharacters == chunkCharacters && previous.OverlapCharacters == overlapCharacters;
        var state = new Dictionary<string, (IndexedFile File, List<IndexedChunk> Chunks)>(StringComparer.Ordinal);
        if (reusable)
        {
            foreach (var file in previous!.Files)
            {
                state[file.Path] = (file, []);
            }

            foreach (var chunk in previous.Chunks)
            {
                if (state.TryGetValue(Relative(folder, chunk.File), out var entry))
                {
                    entry.Chunks.Add(chunk);
                }
            }
        }

        report(null, "Listing documents");
        var disk = await Task.Run(() => List(folder), cancellationToken).ConfigureAwait(false);
        var present = disk.Select(entry => entry.Relative).ToHashSet(StringComparer.Ordinal);
        int removed = 0;
        foreach (string gone in state.Keys.Where(path => !present.Contains(path)).ToList())
        {
            state.Remove(gone);
            removed++;
        }

        int added = 0, changed = 0, unchanged = 0, failed = 0, sinceCheckpoint = 0;
        var clock = Stopwatch.StartNew();
        var lastCheckpoint = TimeSpan.Zero;
        try
        {
            for (int i = 0; i < disk.Count; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                report((double)i / Math.Max(1, disk.Count), $"{i + 1} of {disk.Count} documents");
                var (path, relative, info) = disk[i];
                bool known = state.TryGetValue(relative, out var old);
                if (known && old.File.Size == info.Length && old.File.ModifiedTicks == info.LastWriteTimeUtc.Ticks)
                {
                    unchanged++;
                    continue;
                }

                var outcome = await IndexFileAsync(path, relative, info, known ? old : null, embedder, documentPrompt, chunkCharacters, overlapCharacters, cancellationToken).ConfigureAwait(false);
                state[relative] = (outcome.File, outcome.Chunks);
                if (outcome.File.Error.Length > 0)
                {
                    failed++;
                }

                if (!known)
                {
                    added++;
                }
                else if (outcome.Reembedded)
                {
                    changed++;
                }
                else
                {
                    unchanged++;
                }

                sinceCheckpoint++;
                if (checkpoint is not null && (sinceCheckpoint >= CheckpointFiles || clock.Elapsed - lastCheckpoint > s_checkpointInterval))
                {
                    checkpoint(Build(folder, modelId, chunkCharacters, overlapCharacters, state));
                    sinceCheckpoint = 0;
                    lastCheckpoint = clock.Elapsed;
                }
            }
        }
        catch (OperationCanceledException)
        {
            checkpoint?.Invoke(Build(folder, modelId, chunkCharacters, overlapCharacters, state));
            throw;
        }

        var index = Build(folder, modelId, chunkCharacters, overlapCharacters, state);
        return new IndexUpdate(index, added, changed, removed, unchanged, failed);
    }

    private static string Relative(string folder, string path) => Path.GetRelativePath(folder, path).Replace('\\', '/');

    private List<(string Path, string Relative, FileInfo Info)> List(string folder)
    {
        var options = new EnumerationOptions { RecurseSubdirectories = true, IgnoreInaccessible = true, AttributesToSkip = FileAttributes.System | FileAttributes.ReparsePoint };
        return [.. Directory.EnumerateFiles(folder, "*", options)
            .Where(parser.CanParse)
            .Select(path => (Path: path, Relative: Relative(folder, path)))
            .Where(entry => !entry.Relative.Split('/').SkipLast(1).Any(s_skippedFolders.Contains))
            .Order(Comparer<(string Path, string Relative)>.Create((a, b) => string.CompareOrdinal(a.Relative, b.Relative)))
            .Take(MaxFiles)
            .Select(entry => (entry.Path, entry.Relative, new FileInfo(entry.Path)))];
    }

    private async Task<(IndexedFile File, List<IndexedChunk> Chunks, bool Reembedded)> IndexFileAsync(string path, string relative, FileInfo info, (IndexedFile File, List<IndexedChunk> Chunks)? old,
        ITextEmbedder embedder, string? documentPrompt, int chunkCharacters, int overlapCharacters, CancellationToken cancellationToken)
    {
        long ticks = info.LastWriteTimeUtc.Ticks;
        if (info.Length > parser.MaxBytes(path))
        {
            return (new IndexedFile(relative, info.Length, ticks, "", $"Larger than {parser.MaxBytes(path) >> 20} MB"), [], false);
        }

        try
        {
            string hash = await HashAsync(path, cancellationToken).ConfigureAwait(false);
            if (old is { } known && known.File.Hash == hash && known.File.Error.Length == 0)
            {
                return (known.File with { Size = info.Length, ModifiedTicks = ticks }, known.Chunks, false);
            }

            var sections = await Task.Run(() => parser.Parse(path, cancellationToken), cancellationToken).ConfigureAwait(false);
            var pieces = Cut(sections, chunkCharacters, overlapCharacters);
            var chunks = new List<IndexedChunk>(pieces.Count);
            for (int start = 0; start < pieces.Count; start += BatchSize)
            {
                var batch = pieces.Skip(start).Take(BatchSize).ToList();
                string[] texts = [.. batch.Select(piece => piece.Heading.Length > 0 && !piece.Text.StartsWith(piece.Heading, StringComparison.Ordinal) ? piece.Heading + "\n" + piece.Text : piece.Text)];
                var vectors = await Task.Run(() => embedder.Embed(texts, new TextEmbeddingOptions { Prompt = documentPrompt }), cancellationToken).ConfigureAwait(false);
                for (int j = 0; j < batch.Count; j++)
                {
                    chunks.Add(new IndexedChunk(path, batch[j].Start, batch[j].Text, SearchIndex.Normalize(vectors.Vectors[j]), batch[j].Page, batch[j].Line, batch[j].Heading));
                }
            }

            return (new IndexedFile(relative, info.Length, ticks, hash), chunks, true);
        }
        catch (Exception failure) when (failure is IOException or NotSupportedException or UnauthorizedAccessException or InvalidDataException)
        {
            return (new IndexedFile(relative, info.Length, ticks, "", failure.Message), [], false);
        }
    }

    private static List<Piece> Cut(IReadOnlyList<DocumentSection> sections, int chunkCharacters, int overlapCharacters)
    {
        var pieces = new List<Piece>();
        foreach (var section in sections)
        {
            foreach (var chunk in TextChunker.Split(section.Text, chunkCharacters, overlapCharacters))
            {
                int line = section.Line > 0 ? section.Line + section.Text.AsSpan(0, Math.Min(chunk.Start, section.Text.Length)).Count('\n') : 0;
                pieces.Add(new Piece(chunk.Start, chunk.Text, section.Page, line, section.Heading));
                if (pieces.Count >= MaxChunksPerFile)
                {
                    return pieces;
                }
            }
        }

        return pieces;
    }

    private static async Task<string> HashAsync(string path, CancellationToken cancellationToken)
    {
        await using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite, 1 << 16, useAsync: true);
        return Convert.ToHexString(await SHA256.HashDataAsync(stream, cancellationToken).ConfigureAwait(false)).ToLowerInvariant();
    }

    private static SearchIndex Build(string folder, string modelId, int chunkCharacters, int overlapCharacters, Dictionary<string, (IndexedFile File, List<IndexedChunk> Chunks)> state)
    {
        var ordered = state.OrderBy(pair => pair.Key, StringComparer.Ordinal).Select(pair => pair.Value).ToList();
        return new SearchIndex(folder, modelId, ordered.Count(entry => entry.File.Error.Length == 0), [.. ordered.SelectMany(entry => entry.Chunks)])
        {
            Files = [.. ordered.Select(entry => entry.File)],
            ChunkCharacters = chunkCharacters,
            OverlapCharacters = overlapCharacters,
        };
    }

    private sealed record Piece(int Start, string Text, int Page, int Line, string Heading);
}
