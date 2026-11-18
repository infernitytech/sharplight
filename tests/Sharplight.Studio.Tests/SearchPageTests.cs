using Sharplight.Studio.Models;
using Sharplight.Studio.Tests.Fakes;
using Sharplight.Studio.ViewModels.Inference;
using Sharplight.Studio.ViewModels.Search;

namespace Sharplight.Studio.Tests;

/// <summary>The Search page over a fake embedder, and its pieces.</summary>
public sealed class SearchPageTests
{
    private static FakeModel Embedding(TestServices services, out FakeTextEmbedder embedder)
    {
        embedder = new FakeTextEmbedder();
        return services.Runtime.Add(new FakeModel("fake-embed") { Kind = ModelKindCatalog.Embeddings, Embedder = embedder });
    }

    private static string Documents(TestServices services)
    {
        string folder = Path.Combine(services.Directory, "docs");
        Directory.CreateDirectory(Path.Combine(folder, "sub"));
        File.WriteAllText(Path.Combine(folder, "pets.md"), "The cat sleeps. The dog barks. A cat and a cat and another cat.");
        File.WriteAllText(Path.Combine(folder, "sub", "sea.txt"), "The ship left the harbour. The ship engine roared.");
        File.WriteAllText(Path.Combine(folder, "money.txt"), "Tax tax tax. Pay your tax.");
        File.WriteAllBytes(Path.Combine(folder, "photo.bin"), [1, 2, 3]);
        return folder;
    }

    [Fact]
    public async Task IndexingAFolder_ThenSearching_ListsTheClosestPassagesBestFirst()
    {
        using var services = new TestServices();
        Embedding(services, out var embedder);
        var page = new SearchPageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        Assert.True(page.HasModel);
        Assert.False(page.IndexCommand.CanExecute(null));
        Assert.Contains("Choose a folder", page.RunHint, StringComparison.Ordinal);

        page.SetFolder(Documents(services));
        Assert.True(page.IndexCommand.CanExecute(null));
        Assert.False(page.SearchCommand.CanExecute(null));
        page.IndexCommand.Execute(null);
        await page.Queue.Items.Single().Completion.WaitAsync(TestContext.Current.CancellationToken);

        Assert.True(page.HasIndex);
        Assert.Equal("3 documents, 3 passages, fake-embed", page.IndexSummary);
        Assert.All(embedder.Calls, call => Assert.Null(call!.Prompt));   // documents have no document prompt

        page.Query = "where is the ship engine";
        Assert.True(page.SearchCommand.CanExecute(null));
        page.SearchCommand.Execute(null);
        await page.Queue.Items.Last().Completion.WaitAsync(TestContext.Current.CancellationToken);

        Assert.Equal(3, page.Results.Count);
        Assert.Equal(Path.Combine("sub", "sea.txt"), page.Results[0].FileName);
        Assert.True(page.Results[0].Hit.Score > page.Results[1].Hit.Score);
        Assert.Equal("query", embedder.Calls.Last()!.Prompt);
        Assert.Equal(0.949, page.Results[0].Hit.Score, 3);

        var search = page.History.Items[0];
        Assert.Equal("where is the ship engine", search.Title);
        Assert.Equal("fake-embed", search.Record.Model);
        Assert.Equal(["results.json"], Assert.Single(page.Gallery.Items).Files.Select(file => file.Name));

        page.OpenResultCommand.Execute(page.Results[0]);
        Assert.Equal(Path.Combine(page.FolderPath, "sub", "sea.txt"), services.Platform.Opened.Single());
        await page.CopyResultCommand.ExecuteAsync(page.Results[0]);
        Assert.Contains("ship", services.Platform.Copied.Single(), StringComparison.Ordinal);
    }

    [Fact]
    public async Task Reranking_ReordersTheNearestPassages_WithTheRerankersScores_WhenTurnedOn()
    {
        using var services = new TestServices();
        Embedding(services, out _);
        var reranker = new FakeTextReranker();
        var page = new SearchPageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        Assert.False(page.CanRerank);
        Assert.False(page.Parameters.Parameters.Single(parameter => parameter.Key == "rerank").IsVisibleNow);

        services.Runtime.Add(new FakeModel("fake-rerank") { Kind = ModelKindCatalog.Reranker, Reranker = reranker });
        Assert.True(page.CanRerank);
        Assert.True(page.Parameters.Parameters.Single(parameter => parameter.Key == "rerank").IsVisibleNow);
        page.SetFolder(Documents(services));
        page.IndexCommand.Execute(null);
        await page.Queue.Items.Single().Completion.WaitAsync(TestContext.Current.CancellationToken);

        // Off: the embedding order, no reranker call.
        page.Query = "cat";
        page.SearchCommand.Execute(null);
        await page.Queue.Items.Last().Completion.WaitAsync(TestContext.Current.CancellationToken);
        Assert.Equal("pets.md", page.Results[0].FileName);
        Assert.Empty(reranker.Calls);

        // On: the nearest passages are scored by the reranker and listed by its scores.
        page.Panel.Apply(new Dictionary<string, string> { ["rerank"] = "true", ["topk"] = "2" });
        page.Query = "tax";
        page.SearchCommand.Execute(null);
        await page.Queue.Items.Last().Completion.WaitAsync(TestContext.Current.CancellationToken);
        Assert.Equal(3, Assert.Single(reranker.Calls).Count);   // all three passages are candidates (the nearest 20 at most)
        Assert.Equal(2, page.Results.Count);
        Assert.Equal("money.txt", page.Results[0].FileName);
        Assert.Equal(0.667, page.Results[0].Hit.Score, 3);   // "Tax tax tax. Pay your tax.": 4 of 6 words are the query
        Assert.True(page.Results[0].Hit.Score > page.Results[1].Hit.Score);
        Assert.Equal("fake-rerank", page.History.Items[0].Record.Metadata["reranker"]);
    }

    [Fact]
    public async Task TheResultsOfAPastSearch_ComeBackFromTheHistory()
    {
        using var services = new TestServices();
        Embedding(services, out _);
        var page = new SearchPageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        page.SetFolder(Documents(services));
        page.IndexCommand.Execute(null);
        await page.Queue.Items.Single().Completion.WaitAsync(TestContext.Current.CancellationToken);
        page.Query = "tax";
        page.Panel.Parameters.OfType<Sharplight.Studio.ViewModels.Inference.Parameters.NumberParameterViewModel>().First(parameter => parameter.Key == "topk").Value = 2;
        page.SearchCommand.Execute(null);
        await page.Queue.Items.Last().Completion.WaitAsync(TestContext.Current.CancellationToken);
        Assert.Equal(2, page.Results.Count);

        var reopened = new SearchPageViewModel(services.Services, new FakeNavigator());
        Assert.Equal(2, reopened.Panel.Options.TopK);
        reopened.History.RestoreCommand.Execute(reopened.History.Items.First(item => item.Title == "tax"));
        Assert.Equal("tax", reopened.Query);
        Assert.Equal(2, reopened.Results.Count);
        Assert.Contains("money.txt", reopened.Results[0].FileName, StringComparison.Ordinal);
        Assert.False(reopened.HasIndex);   // the index itself lives in memory
    }

    [Fact]
    public async Task AnIndexIsOnlyUsedWithItsOwnModel()
    {
        using var services = new TestServices();
        Embedding(services, out _);
        services.Runtime.Add(new FakeModel("other-embed") { Kind = ModelKindCatalog.Embeddings, Embedder = new FakeTextEmbedder() });
        var page = new SearchPageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        page.SetFolder(Documents(services));
        page.IndexCommand.Execute(null);
        await page.Queue.Items.Single().Completion.WaitAsync(TestContext.Current.CancellationToken);

        page.Picker.SelectedModel = page.Picker.Models.Single(option => option.Id == "other-embed");
        page.Query = "cat";
        Assert.Contains("index the folder again", page.RunHint, StringComparison.Ordinal);
        page.SearchCommand.Execute(null);
        Assert.Contains("fake-embed", page.Notice, StringComparison.Ordinal);
        Assert.Single(page.Queue.Items);
    }

    [Fact]
    public async Task AFailingEmbedder_IsReportedOnTheRun()
    {
        using var services = new TestServices();
        services.Runtime.Add(new FakeModel("broken") { Kind = ModelKindCatalog.Embeddings });   // no embedder: the lease fails
        var page = new SearchPageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        page.SetFolder(Documents(services));
        page.IndexCommand.Execute(null);
        await page.Queue.Items.Single().Completion.WaitAsync(TestContext.Current.CancellationToken);
        Assert.Equal(RunStatus.Failed, page.Queue.Items[0].Status);
        Assert.Contains("not an embedding model", page.Notice, StringComparison.Ordinal);
        Assert.False(page.HasIndex);
    }

    [Fact]
    public void TextChunker_CutsAtParagraphsAndSentences_WithOverlap()
    {
        string text = string.Join("\n\n", Enumerable.Range(1, 6).Select(i => $"Paragraph {i}. " + new string('x', 80)));
        var chunks = TextChunker.Split(text, 200, 40);
        Assert.True(chunks.Count >= 3);
        Assert.All(chunks, chunk => Assert.InRange(chunk.Text.Length, 1, 200));
        Assert.StartsWith("Paragraph 1.", chunks[0].Text, StringComparison.Ordinal);
        Assert.True(chunks[1].Start < chunks[0].Start + chunks[0].Text.Length + 1);   // overlap
        Assert.Contains("Paragraph 6.", chunks[^1].Text, StringComparison.Ordinal);
        Assert.Empty(TextChunker.Split("   ", 200, 0));
        Assert.Single(TextChunker.Split("short", 200, 0));
        Assert.True(TextChunker.Split(new string('y', 1000), 100, 20).Count >= 10);   // no break points: hard cut
    }

    [Fact]
    public void SearchIndex_RanksByCosineSimilarity()
    {
        var index = new SearchIndex("/d", "m", 2,
        [
            new IndexedChunk("/d/a", 0, "a", SearchIndex.Normalize([1, 0])),
            new IndexedChunk("/d/b", 0, "b", SearchIndex.Normalize([1, 1])),
            new IndexedChunk("/d/c", 0, "c", SearchIndex.Normalize([0, 5])),
        ]);
        var nearest = index.Nearest([3, 0.1f], 2);
        Assert.Equal(["a", "b"], nearest.Select(hit => hit.Chunk.Text));
        Assert.Equal(0.9994, nearest[0].Score, 3);
        Assert.Equal([0f, 0f], SearchIndex.Normalize([0, 0]));
    }
}
