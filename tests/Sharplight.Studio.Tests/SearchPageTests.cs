using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Runtime;
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
        await reopened.StoredIndexLoad;
        Assert.True(reopened.HasIndex);   // the stored index of the restored folder is loaded again
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

    private static async Task IndexedAsync(SearchPageViewModel page, string folder)
    {
        page.SetFolder(folder);
        page.IndexCommand.Execute(null);
        await page.Queue.Items.Last().Completion.WaitAsync(TestContext.Current.CancellationToken);
    }

    private static async Task RunAsync(SearchPageViewModel page, System.Windows.Input.ICommand command)
    {
        command.Execute(null);
        await page.Queue.Items.Last().Completion.WaitAsync(TestContext.Current.CancellationToken);
    }

    [Fact]
    public async Task TheIndex_IsStored_AndComesBackWhenTheFolderIsChosenAgain()
    {
        using var services = new TestServices();
        Embedding(services, out var embedder);
        string folder = Documents(services);
        var page = new SearchPageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        await IndexedAsync(page, folder);
        Assert.Single(Directory.GetFiles(Path.Combine(services.Directory, "pages", "search", "indexes"), "*.slsx"));

        // A new page (a restart): choosing the folder loads the stored index, so a search needs no indexing and no re-embedding.
        embedder.Texts.Clear();
        var reopened = new SearchPageViewModel(services.Services, new FakeNavigator());
        await reopened.ActivateAsync();
        reopened.SetFolder(folder);
        await reopened.StoredIndexLoad;
        Assert.True(reopened.HasIndex);
        Assert.Equal("3 documents, 3 passages, fake-embed", reopened.IndexSummary);
        reopened.Query = "tax";
        await RunAsync(reopened, reopened.SearchCommand);
        Assert.Equal("money.txt", reopened.Results[0].FileName);
        Assert.Equal(["tax"], embedder.Texts);   // only the question was embedded

        // Another folder starts without an index.
        reopened.SetFolder(Path.Combine(services.Directory, "docs", "sub"));
        await reopened.StoredIndexLoad;
        Assert.False(reopened.HasIndex);
    }

    [Fact]
    public async Task IndexingAgain_UpdatesOnlyWhatChanged_AndAnotherModelOffersAReindex()
    {
        using var services = new TestServices();
        Embedding(services, out var embedder);
        services.Runtime.Add(new FakeModel("other-embed") { Kind = ModelKindCatalog.Embeddings, Embedder = new FakeTextEmbedder() });
        string folder = Documents(services);
        var page = new SearchPageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        await IndexedAsync(page, folder);
        Assert.Equal("3", page.History.Items[0].Record.Metadata["added"]);

        File.WriteAllText(Path.Combine(folder, "money.txt"), "Tax tax. A new ship tax.");
        File.WriteAllText(Path.Combine(folder, "new.txt"), "A brand new cat.");
        File.Delete(Path.Combine(folder, "pets.md"));
        embedder.Texts.Clear();
        await RunAsync(page, page.IndexCommand);
        var metadata = page.History.Items[0].Record.Metadata;
        Assert.Equal(("1", "1", "1"), (metadata["added"], metadata["changed"], metadata["removed"]));
        Assert.Equal(2, embedder.Texts.Count);   // the changed and the added file only
        Assert.Equal("3 documents, 3 passages, fake-embed", page.IndexSummary);

        page.Picker.SelectedModel = page.Picker.Models.Single(option => option.Id == "other-embed");
        Assert.Contains("index the folder again", page.RunHint, StringComparison.Ordinal);
        await RunAsync(page, page.IndexCommand);
        Assert.Equal("3", page.History.Items[0].Record.Metadata["added"]);   // a model change embeds everything again
        Assert.Equal("3 documents, 3 passages, other-embed", page.IndexSummary);
    }

    [Fact]
    public async Task KeywordSearch_NeedsNoEmbedding_AndHybridCombinesBothRankings()
    {
        using var services = new TestServices();
        Embedding(services, out var embedder);
        string folder = Path.Combine(services.Directory, "docs");
        Directory.CreateDirectory(folder);
        File.WriteAllText(Path.Combine(folder, "codes.txt"), "Error code E1042 means the disk is full.");
        File.WriteAllText(Path.Combine(folder, "cats.txt"), "The cat sleeps. A cat and a cat.");
        File.WriteAllText(Path.Combine(folder, "sea.txt"), "The ship engine roared at the ship.");
        var page = new SearchPageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        await IndexedAsync(page, folder);

        page.Panel.Apply(new Dictionary<string, string> { ["mode"] = "keyword", ["topk"] = "5" });
        page.Query = "E1042";
        int calls = embedder.Calls.Count;
        await RunAsync(page, page.SearchCommand);
        Assert.Equal(calls, embedder.Calls.Count);   // BM25 alone does not touch the model
        Assert.Equal(["codes.txt"], page.Results.Select(result => result.FileName));
        Assert.Equal(1.0, page.Results[0].Hit.Score, 6);
        Assert.Equal("keyword", page.History.Items[0].Record.Metadata["mode"]);
        Assert.Equal(["E1042"], page.Results[0].Segments.Where(segment => segment.IsMatch).Select(segment => segment.Text));

        // The embedder knows only five topic words, so "E1042" means nothing to it; the keyword ranking brings the code in.
        page.Panel.Apply(new Dictionary<string, string> { ["mode"] = "hybrid" });
        await RunAsync(page, page.SearchCommand);
        Assert.Equal("codes.txt", page.Results[0].FileName);
        Assert.Equal((1.0 / 61 + 1.0 / 62) / (2.0 / 61), page.Results[0].Hit.Score, 6);   // first by keywords, third of three (a zero-similarity tie) by embedding
        page.Query = "cat";
        await RunAsync(page, page.SearchCommand);
        Assert.Equal("cats.txt", page.Results[0].FileName);
        Assert.Equal(1.0, page.Results[0].Hit.Score, 6);
        Assert.InRange(page.Results[1].Hit.Score, 0.0, 1.0);
        Assert.Equal("hybrid", page.History.Items[0].Record.Metadata["mode"]);
    }

    [Fact]
    public async Task Results_NameTheirPlace_AndOpenAtIt()
    {
        using var services = new TestServices();
        Embedding(services, out _);
        string folder = Path.Combine(services.Directory, "docs");
        Directory.CreateDirectory(folder);
        File.WriteAllText(Path.Combine(folder, "guide.md"), "# Intro\nWelcome.\n\n# Ships\nThe ship engine roared.\n");
        DocumentFixtures.Pdf(Path.Combine(folder, "book.pdf"), "Nothing here.", "The cat sat.");
        var page = new SearchPageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        await IndexedAsync(page, folder);

        page.Query = "ship engine";
        await RunAsync(page, page.SearchCommand);
        var ship = page.Results.Single(result => result.FileName == "guide.md" && result.Hit.Heading == "Ships");
        Assert.Equal("line 4 · Ships", ship.Location);
        page.OpenResultCommand.Execute(ship);
        Assert.Equal((Path.Combine(folder, "guide.md"), 0, 4), services.Platform.OpenedAt.Single());

        page.Query = "cat";
        await RunAsync(page, page.SearchCommand);
        var cat = page.Results[0];
        Assert.Equal("book.pdf", cat.FileName);
        Assert.Equal("page 2", cat.Location);
        page.OpenResultCommand.Execute(cat);
        Assert.Equal((Path.Combine(folder, "book.pdf"), 2, 0), services.Platform.OpenedAt.Last());
    }

    [Fact]
    public async Task Ask_AnswersFromTheTopPassages_WithCitationsLinkedToTheResults()
    {
        using var services = new TestServices();
        Embedding(services, out _);
        var page = new SearchPageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        Assert.False(page.CanAskModel);
        await IndexedAsync(page, Documents(services));
        page.Query = "where is the ship engine";
        Assert.False(page.AskCommand.CanExecute(null));   // no chat model yet

        var chat = services.Runtime.Add(new FakeModel("fake-chat"));
        chat.Script.Clear();
        chat.Script.AddRange([ChatStreamUpdate.Content("The ship engine is in "), ChatStreamUpdate.Content("the harbour notes [1], "), ChatStreamUpdate.Content("see also [2] and [9]."),
            ChatStreamUpdate.Finished(new MessageStats { PromptTokens = 50, CompletionTokens = 20 })]);
        Assert.True(page.CanAskModel);
        Assert.True(page.AskCommand.CanExecute(null));
        await RunAsync(page, page.AskCommand);

        Assert.Equal("The ship engine is in the harbour notes [1], see also [2] and [9].", page.Answer);
        Assert.True(page.HasAnswer);
        Assert.Equal(3, page.Results.Count);
        Assert.Equal([page.Results[0], page.Results[1]], page.AnswerCitations);   // [9] points to no passage and is not linked
        Assert.Equal("[1]", page.Results[0].NumberText);
        page.FocusResultCommand.Execute(page.AnswerCitations[1]);
        Assert.Equal([false, true, false], page.Results.Select(result => result.IsFocused));

        // The prompt carries the numbered passages and the question; the answer is not asked to think.
        var input = Assert.Single(chat.Inputs);
        Assert.False(input.EnableThinking);
        Assert.Equal(ChatRole.System, input.Messages[0].Role);
        Assert.Contains("[1] sea.txt, line 1", input.Messages[1].Text, StringComparison.Ordinal);
        Assert.Contains("The ship engine roared.", input.Messages[1].Text, StringComparison.Ordinal);
        Assert.Contains("Question: where is the ship engine", input.Messages[1].Text, StringComparison.Ordinal);

        var run = page.History.Items[0];
        Assert.StartsWith("Ask: ", run.Title, StringComparison.Ordinal);
        Assert.Equal("fake-chat", run.Record.Metadata["answered by"]);
        var reopened = new SearchPageViewModel(services.Services, new FakeNavigator());
        reopened.History.RestoreCommand.Execute(reopened.History.Items.First(item => item.Title.StartsWith("Ask: ", StringComparison.Ordinal)));
        Assert.Equal(page.Answer, reopened.Answer);
        Assert.Equal(2, reopened.AnswerCitations.Count);

        // A new search clears the old answer.
        page.Query = "tax";
        await RunAsync(page, page.SearchCommand);
        Assert.False(page.HasAnswer);
        Assert.Empty(page.AnswerCitations);
    }

    [Fact]
    public async Task Ask_RunsOffTheCallingThread_AndAFailingChatModelIsReported()
    {
        using var services = new TestServices();
        Embedding(services, out _);
        var chat = services.Runtime.Add(new FakeModel("fake-chat") { Gate = new TaskCompletionSource() });
        var page = new SearchPageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        await IndexedAsync(page, Documents(services));
        page.Query = "cat";
        page.AskCommand.Execute(null);   // returns at once: the work is a run of the queue, parked on the model's gate
        Assert.True(page.IsWorking);
        page.CancelCommand.Execute(null);
        await page.Queue.Items.Last().Completion.WaitAsync(TestContext.Current.CancellationToken).ContinueWith(_ => { }, TestContext.Current.CancellationToken);
        Assert.Equal(RunStatus.Cancelled, page.Queue.Items.Last().Status);

        chat.Gate = null;
        chat.Script.Clear();
        chat.Script.Add(ChatStreamUpdate.Finished(new MessageStats(), "out of memory"));
        page.AskCommand.Execute(null);
        await page.Queue.Items.Last().Completion.WaitAsync(TestContext.Current.CancellationToken).ContinueWith(_ => { }, TestContext.Current.CancellationToken);
        Assert.Equal(RunStatus.Failed, page.Queue.Items.Last().Status);
        Assert.Contains("could not answer", page.Notice, StringComparison.Ordinal);
    }

    [Fact]
    public async Task WatchingTheFolder_UpdatesTheIndexIncrementally_WhenChangesAreReported()
    {
        using var services = new TestServices();
        Embedding(services, out var embedder);
        string folder = Documents(services);
        var page = new SearchPageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        await IndexedAsync(page, folder);

        // Not watching: a change report does nothing.
        File.WriteAllText(Path.Combine(folder, "late.txt"), "A late cat.");
        int runs = page.Queue.Items.Count;
        page.NotifyFolderChanged();
        Assert.Equal(runs, page.Queue.Items.Count);

        page.WatchFolder = true;
        embedder.Texts.Clear();
        page.NotifyFolderChanged();
        await page.Queue.Items.Last().Completion.WaitAsync(TestContext.Current.CancellationToken);
        Assert.Equal(runs + 1, page.Queue.Items.Count);
        Assert.Equal(["A late cat."], embedder.Texts);
        Assert.Equal("4 documents, 4 passages, fake-embed", page.IndexSummary);
        page.WatchFolder = false;
        await page.ActivateAsync();
        page.Deactivate();
    }

    [Fact]
    public async Task TheFolderWatcher_CallsBackOnceAfterABurstOfChanges()
    {
        using var directory = new TempDir();
        int calls = 0;
        using var watcher = new Sharplight.Studio.Services.Search.FolderWatcher(directory.Path, TimeSpan.FromMilliseconds(300), () => Interlocked.Increment(ref calls));
        await Task.Delay(100, TestContext.Current.CancellationToken);
        for (int i = 0; i < 5; i++)
        {
            File.WriteAllText(directory.File($"f{i}.txt"), "x");
            await Task.Delay(30, TestContext.Current.CancellationToken);
        }

        for (int wait = 0; wait < 100 && Volatile.Read(ref calls) == 0; wait++)
        {
            await Task.Delay(50, TestContext.Current.CancellationToken);
        }

        await Task.Delay(500, TestContext.Current.CancellationToken);
        Assert.Equal(1, Volatile.Read(ref calls));
    }
}
