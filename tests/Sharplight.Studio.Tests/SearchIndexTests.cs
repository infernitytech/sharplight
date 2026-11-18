using Sharplight.Studio.Services.Search;
using Sharplight.Studio.Tests.Fakes;
using Sharplight.Studio.ViewModels.Search;

namespace Sharplight.Studio.Tests;

/// <summary>The stored index and the incremental indexer: what is re-embedded when files change, and resuming after a cancel.</summary>
public sealed class SearchIndexTests : IDisposable
{
    private readonly TempDir _dir = new();
    private readonly FakeTextEmbedder _embedder = new();

    public void Dispose() => _dir.Dispose();

    private string Folder => _dir.File("docs");

    private string Write(string name, string text, int minute = 0)
    {
        string path = Path.Combine(Folder, name);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, text);
        File.SetLastWriteTimeUtc(path, new DateTime(2026, 1, 1, 0, minute, 0, DateTimeKind.Utc));
        return path;
    }

    private Task<IndexUpdate> RunAsync(SearchIndex? previous, string model = "m", int chunk = 1000, IDocumentParser? parser = null, Action<double?, string>? report = null,
        Action<SearchIndex>? checkpoint = null, CancellationToken cancellation = default)
    {
        return new SearchIndexer(parser ?? new StructuredDocumentParser()).UpdateAsync(Folder, previous, model, _embedder, null, chunk, 100, report ?? ((_, _) => { }), checkpoint, cancellation);
    }

    private void Corpus()
    {
        Write("a.md", "# Cats\nThe cat sleeps all day.\n\n# Dogs\nThe dog barks at the ship.\n");
        Write("b.txt", "The ship engine roared.");
        Write("sub/c.txt", "Pay your tax in April.");
    }

    [Fact]
    public async Task TheIndex_RoundTripsThroughItsFile_WithHalfPrecisionVectors()
    {
        Corpus();
        var update = await RunAsync(null);
        string file = SearchIndexStore.PathFor(_dir.File("pages/search"), Folder);
        SearchIndexStore.Save(file, update.Index);
        Assert.False(File.Exists(file + ".tmp"));

        var loaded = SearchIndexStore.Load(file)!;
        Assert.Equal(update.Index.Folder, loaded.Folder);
        Assert.Equal("m", loaded.ModelId);
        Assert.Equal(3, loaded.FileCount);
        Assert.Equal((1000, 100), (loaded.ChunkCharacters, loaded.OverlapCharacters));
        Assert.Equal(update.Index.Files, loaded.Files);
        Assert.Equal(update.Index.Chunks.Count, loaded.Chunks.Count);
        for (int i = 0; i < loaded.Chunks.Count; i++)
        {
            var (a, b) = (update.Index.Chunks[i], loaded.Chunks[i]);
            Assert.Equal((a.File, a.Start, a.Text, a.Page, a.Line, a.Heading), (b.File, b.Start, b.Text, b.Page, b.Line, b.Heading));
            Assert.Equal(a.Vector, b.Vector, (x, y) => Math.Abs(x - y) < 1e-3);
        }

        Assert.Equal(["Cats", "Dogs"], loaded.Chunks.Where(chunk => chunk.File.EndsWith("a.md", StringComparison.Ordinal)).Select(chunk => chunk.Heading));
        Assert.Equal([1, 4], loaded.Chunks.Where(chunk => chunk.File.EndsWith("a.md", StringComparison.Ordinal)).Select(chunk => chunk.Line));
        Assert.Equal(SearchIndexStore.PathFor("/p", Folder + "/"), SearchIndexStore.PathFor("/p", Folder));
        Assert.NotEqual(SearchIndexStore.PathFor("/p", Folder), SearchIndexStore.PathFor("/p", Folder + "2"));
    }

    [Fact]
    public async Task ADamagedOrMissingIndexFile_LoadsAsNothing()
    {
        string file = _dir.File("x.slsx");
        Assert.Null(SearchIndexStore.Load(file));
        File.WriteAllBytes(file, [1, 2, 3, 4, 5, 6, 7, 8]);
        Assert.Null(SearchIndexStore.Load(file));
        Corpus();
        SearchIndexStore.Save(file, (await RunAsync(null)).Index);
        byte[] bytes = File.ReadAllBytes(file);
        File.WriteAllBytes(file, bytes[..(bytes.Length / 2)]);
        Assert.Null(SearchIndexStore.Load(file));
    }

    [Fact]
    public async Task AnUpdate_EmbedsOnlyChangedAndAddedFiles_AndDropsRemovedOnes()
    {
        Corpus();
        var first = await RunAsync(null);
        Assert.Equal((3, 0, 0, 0), (first.Added, first.Changed, first.Removed, first.Unchanged));
        int embedded = _embedder.Texts.Count;
        Assert.Equal(first.Index.Chunks.Count, embedded);

        // Nothing changed: nothing is read or embedded.
        var same = await RunAsync(first.Index);
        Assert.Equal((0, 0, 0, 3), (same.Added, same.Changed, same.Removed, same.Unchanged));
        Assert.Equal(embedded, _embedder.Texts.Count);

        // b changes, c disappears, d appears, a is only touched (same content, new time).
        Write("b.txt", "The harbour was quiet.", minute: 5);
        File.Delete(Path.Combine(Folder, "sub", "c.txt"));
        Write("d.txt", "The dog chased the cat.");
        Write("a.md", File.ReadAllText(Path.Combine(Folder, "a.md")), minute: 9);
        _embedder.Texts.Clear();
        var update = await RunAsync(same.Index);
        Assert.Equal((1, 1, 1, 1), (update.Added, update.Changed, update.Removed, update.Unchanged));
        Assert.Equal(["The harbour was quiet.", "The dog chased the cat."], _embedder.Texts.Order().Reverse());
        Assert.Equal(["a.md", "b.txt", "d.txt"], update.Index.Files.Select(file => file.Path));
        Assert.DoesNotContain(update.Index.Chunks, chunk => chunk.File.EndsWith("c.txt", StringComparison.Ordinal));
        Assert.Contains(update.Index.Chunks, chunk => chunk.Text == "The harbour was quiet.");
        Assert.DoesNotContain(update.Index.Chunks, chunk => chunk.Text == "The ship engine roared.");
        Assert.Equal(new DateTime(2026, 1, 1, 0, 9, 0, DateTimeKind.Utc).Ticks, update.Index.Files[0].ModifiedTicks);   // the touched file's new time is remembered
    }

    [Fact]
    public async Task AnotherModelOrPassageSetting_IndexesEverythingAgain()
    {
        Corpus();
        var first = await RunAsync(null, "m");
        _embedder.Texts.Clear();
        Assert.Equal(3, (await RunAsync(first.Index, "other")).Added);
        Assert.Equal(3, (await RunAsync(first.Index, "m", chunk: 500)).Added);
        Assert.Equal(first.Index.Chunks.Count * 2, _embedder.Texts.Count);
    }

    [Fact]
    public async Task ACancelledRun_KeepsItsProgress_AndTheNextRunResumesAfterIt()
    {
        for (int i = 0; i < 5; i++)
        {
            Write($"f{i}.txt", $"document number {i} about the ship", minute: i);
        }

        using var cancel = new CancellationTokenSource();
        SearchIndex? saved = null;
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => RunAsync(null, report: (_, text) =>
        {
            if (text.StartsWith("3 of 5", StringComparison.Ordinal))
            {
                cancel.Cancel();
            }
        }, checkpoint: index => saved = index, cancellation: cancel.Token));
        Assert.Equal(["f0.txt", "f1.txt"], saved!.Files.Select(file => file.Path));
        Assert.Equal(2, _embedder.Texts.Count);

        _embedder.Texts.Clear();
        var resumed = await RunAsync(saved);
        Assert.Equal((3, 0, 0, 2), (resumed.Added, resumed.Changed, resumed.Removed, resumed.Unchanged));
        Assert.Equal(3, _embedder.Texts.Count);
        Assert.Equal(5, resumed.Index.FileCount);
    }

    [Fact]
    public async Task AFileOverTheLimit_IsRecordedWithItsReason_AndNotRetriedUntilItChanges()
    {
        Write("small.txt", "fine");
        Write("huge.txt", new string('x', 3000));
        var parser = new StructuredDocumentParser(new DocumentLimits { MaxTextBytes = 1000 });
        var first = await RunAsync(null, parser: parser);
        Assert.Equal(1, first.Failed);
        Assert.Equal(1, first.Index.FileCount);   // only the readable one counts
        Assert.Contains("Larger than", first.Index.Files.Single(file => file.Path == "huge.txt").Error, StringComparison.Ordinal);

        var again = await RunAsync(first.Index, parser: parser);
        Assert.Equal((0, 0, 2), (again.Added, again.Failed, again.Unchanged));

        Write("huge.txt", "now small", minute: 3);
        var fixedUp = await RunAsync(again.Index, parser: parser);
        Assert.Equal(0, fixedUp.Failed);
        Assert.Equal(2, fixedUp.Index.FileCount);
    }

    [Fact]
    public async Task AFolderIsListedWithoutVersionControlAndDependencyFolders()
    {
        Write("a.txt", "kept");
        Write(".git/config.txt", "hidden");
        Write("node_modules/pkg/readme.txt", "dependency");
        var update = await RunAsync(null);
        Assert.Equal(["a.txt"], update.Index.Files.Select(file => file.Path));
    }

    [Fact]
    public async Task EveryFormatIsIndexedWithItsPositions()
    {
        Write("n.md", "# Notes\nship notes");
        DocumentFixtures.Pdf(Path.Combine(Folder, "p.pdf"), "first page", "second page about the engine");
        DocumentFixtures.Docx(Path.Combine(Folder, "d.docx"), "#Plan", "tax plan");
        DocumentFixtures.Pptx(Path.Combine(Folder, "s.pptx"), "Intro\ncat slide", "Outro\ndog slide");
        DocumentFixtures.Xlsx(Path.Combine(Folder, "x.xlsx"), ("Data", [["ship", "#4"]]));
        Write("h.html", "<h1>Title</h1><p>web ship</p>");
        Write("code.cs", "class A\n{\n}\n\n\nclass Ship\n{\n}\n");
        var update = await RunAsync(null, chunk: 200);
        Assert.Equal(7, update.Index.FileCount);
        var pdf = update.Index.Chunks.Where(chunk => chunk.File.EndsWith(".pdf", StringComparison.Ordinal)).ToList();
        Assert.Equal([1, 2], pdf.Select(chunk => chunk.Page));
        Assert.Equal([1, 2], update.Index.Chunks.Where(chunk => chunk.File.EndsWith(".pptx", StringComparison.Ordinal)).Select(chunk => chunk.Page));
        Assert.Equal("Data", update.Index.Chunks.Single(chunk => chunk.File.EndsWith(".xlsx", StringComparison.Ordinal)).Heading);
        Assert.Equal(1, update.Index.Chunks.Single(chunk => chunk.File.EndsWith("code.cs", StringComparison.Ordinal)).Line);
        // Keyword search over every format finds the "ship" passages.
        Assert.Equal(4, update.Index.Keyword("ship", 10).Count);
    }
}
