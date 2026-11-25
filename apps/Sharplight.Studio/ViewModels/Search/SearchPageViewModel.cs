using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Llm.Embeddings;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services;
using Sharplight.Studio.Services.Pages;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.Services.Search;
using Sharplight.Studio.ViewModels.Inference;
using Sharplight.Studio.ViewModels.Inference.Parameters;

namespace Sharplight.Studio.ViewModels.Search;

/// <summary>
/// The Search page (kind <c>embeddings</c>): a folder of documents is read (PDF, Office, HTML, Markdown, text and source), cut into
/// passages along their structure and embedded by the chosen text embedding model. The index is stored for that model and updated
/// incrementally (only changed, added and removed files are processed). A question is answered with the closest passages by embedding
/// similarity, BM25 keywords or both (reciprocal rank fusion), optionally reranked, and with a resident chat model <c>Ask</c> writes an
/// answer from the top passages with citations. Built from the shared inference blocks (picker, parameters, queue, history, outputs); both indexing and searching are runs
/// of the queue, so the embedding model's work never touches the UI thread and can be cancelled.
/// </summary>
internal sealed partial class SearchPageViewModel : InferencePageViewModel
{
    private const string ResultsName = "results.json";
    private const int CandidatePool = 50;

    private readonly IDocumentParser _parser;
    private readonly SearchIndexer _indexer;
    private SearchIndex? _index;
    private FolderWatcher? _watcher;
    private bool _changedWhileBusy;
    private bool _active;

    /// <param name="services">The services.</param>
    /// <param name="navigator">Moves to the library and Discover.</param>
    /// <param name="parser">Reads the documents; the default reads every supported format.</param>
    public SearchPageViewModel(StudioServices services, INavigator navigator, IDocumentParser? parser = null)
        : base(services, navigator, ModelKindCatalog.Embeddings, "search", CreateParameters(services))
    {
        _parser = parser ?? new StructuredDocumentParser();
        _indexer = new SearchIndexer(_parser);
        ChatModels = new ModelPickerViewModel(services, Sharplight.Models.ModelKinds.TextGeneration);
        ChatModels.PropertyChanged += (_, args) =>
        {
            if (args.PropertyName is nameof(ModelPickerViewModel.HasModels) or nameof(ModelPickerViewModel.SelectedModel))
            {
                OnInputChanged();
            }
        };
        Rerankers = new ModelPickerViewModel(services, ModelKindCatalog.Reranker);
        Rerankers.PropertyChanged += (_, args) =>
        {
            if (args.PropertyName == nameof(ModelPickerViewModel.HasModels))
            {
                UpdateRerankers();
            }
        };
        UpdateRerankers();
    }

    /// <summary>The resident reranking models and the chosen one (used when <c>Rerank results</c> is on).</summary>
    public ModelPickerViewModel Rerankers { get; }

    /// <summary>The resident chat models and the chosen one, which writes the answer of <c>Ask</c>.</summary>
    public ModelPickerViewModel ChatModels { get; }

    /// <summary>A chat model is resident, so a question can be answered from the passages.</summary>
    public bool CanAskModel => ChatModels.HasModels;

    /// <summary>A reranking model is resident, so results can be reranked.</summary>
    public bool CanRerank => Rerankers.HasModels;

    /// <inheritdoc/>
    public override string Title => "Search";

    /// <inheritdoc/>
    public override string Icon => "IconSearch";

    /// <inheritdoc/>
    public override string KindNoun => "text embeddings";

    /// <summary>The parameter panel, typed.</summary>
    public ParameterPanelViewModel<SearchOptions> Panel => (ParameterPanelViewModel<SearchOptions>)Parameters;

    /// <summary>The passages the last search found, best first.</summary>
    public ObservableCollection<SearchResultViewModel> Results { get; } = [];

    /// <summary>The folder to index.</summary>
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(IndexCommand))]
    public partial string FolderPath { get; set; } = "";

    /// <summary>The question.</summary>
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(SearchCommand))]
    public partial string Query { get; set; } = "";

    /// <summary>What the index holds (files, passages, model), or empty.</summary>
    [ObservableProperty]
    public partial string IndexSummary { get; set; } = "";

    /// <summary>What the last search did (passages, time), or empty.</summary>
    [ObservableProperty]
    public partial string ResultSummary { get; set; } = "";

    /// <summary>The model's answer to the last <c>Ask</c>, with <c>[n]</c> citations of the passages below, or empty.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasAnswer))]
    public partial string Answer { get; set; } = "";

    /// <summary>The answer is being written or is shown.</summary>
    public bool HasAnswer => Answer.Length > 0;

    /// <summary>The passages the answer cites, in order of first mention.</summary>
    public ObservableCollection<SearchResultViewModel> AnswerCitations { get; } = [];

    /// <summary>Update the index by itself when files in the folder change while the page is open.</summary>
    [ObservableProperty]
    public partial bool WatchFolder { get; set; }

    /// <summary>The load of the stored index the last folder change started (tests and callers can wait for it).</summary>
    public Task StoredIndexLoad { get; private set; } = Task.CompletedTask;

    /// <summary>A folder is indexed.</summary>
    public bool HasIndex => _index is not null;

    /// <summary>A search ran and found passages.</summary>
    public bool HasResults => Results.Count > 0;

    /// <summary>A search is running or queued, or the folder is being indexed.</summary>
    public bool IsWorking => Queue.IsBusy;

    /// <summary>What to do next, for the hint under the buttons.</summary>
    public string RunHint => !HasModel ? "Load a text embedding model first."
        : FolderPath.Length == 0 ? "Choose a folder of documents to index."
        : !HasIndex ? "Index the folder, then ask a question."
        : _index!.ModelId != SelectedModel?.Id ? $"The index was made with {_index.ModelId}; index the folder again for this model."
        : "";

    /// <summary>Indexes <paramref name="folder"/> (as if chosen).</summary>
    /// <param name="folder">The folder.</param>
    public void SetFolder(string folder) => FolderPath = folder;

    /// <inheritdoc/>
    public override Task ActivateAsync()
    {
        Rerankers.Refresh();
        ChatModels.Refresh();
        _active = true;
        UpdateWatcher();
        if (WatchFolder)
        {
            NotifyFolderChanged();
        }

        return base.ActivateAsync();
    }

    /// <inheritdoc/>
    public override void Deactivate()
    {
        _active = false;
        UpdateWatcher();
        base.Deactivate();
    }

    /// <summary>
    /// Brings the stored index up to date after files of the folder changed (called by the watcher once changes are quiet): runs an
    /// incremental index when a stored index of the chosen model exists and nothing else is running, else remembers it for later.
    /// </summary>
    public void NotifyFolderChanged()
    {
        if (!WatchFolder || _index is not { } index || SelectedModel is not { } model || index.ModelId != model.Id)
        {
            return;
        }

        if (Queue.IsBusy)
        {
            _changedWhileBusy = true;
            return;
        }

        _changedWhileBusy = false;
        Index();
    }

    partial void OnWatchFolderChanged(bool value) => UpdateWatcher();

    private void UpdateWatcher()
    {
        _watcher?.Dispose();
        _watcher = null;
        if (_active && WatchFolder && Directory.Exists(FolderPath))
        {
            _watcher = new FolderWatcher(FolderPath, TimeSpan.FromSeconds(2), () => Services.Dispatcher.Post(NotifyFolderChanged));
        }
    }

    private void UpdateRerankers()
    {
        Panel.SetVisible("rerank", Rerankers.HasModels);
        OnPropertyChanged(nameof(CanRerank));
    }

    /// <inheritdoc/>
    protected override void OnModelChanged(ILoadedModel? model) => OnInputChanged();

    /// <inheritdoc/>
    protected override void OnBusyChanged()
    {
        OnPropertyChanged(nameof(IsWorking));
        IndexCommand.NotifyCanExecuteChanged();
        SearchCommand.NotifyCanExecuteChanged();
        AskCommand.NotifyCanExecuteChanged();
        if (!Queue.IsBusy && _changedWhileBusy)
        {
            NotifyFolderChanged();
        }
    }

    /// <inheritdoc/>
    protected override void Restore(RunRecord record)
    {
        string path = Archive.PathOf(record, ResultsName);
        if (!File.Exists(path))
        {
            if (record.Input.TryGetValue("folder", out var indexed))
            {
                FolderPath = indexed;
            }

            return;
        }

        try
        {
            var file = JsonSerializer.Deserialize(File.ReadAllBytes(path), StudioJsonContext.Default.SearchResultsFile);
            if (file is null)
            {
                return;
            }

            Query = file.Query;
            if (file.Folder.Length > 0)
            {
                FolderPath = file.Folder;
            }

            Show(file.Hits, file.Folder, file.Query);
            ShowAnswer(file.Answer);
            ResultSummary = string.Join(" · ", record.Metadata.Select(pair => $"{pair.Key}: {pair.Value}"));
            Parameters.Apply(record.Options);
        }
        catch (Exception error) when (error is IOException or JsonException or UnauthorizedAccessException)
        {
            Notice = $"The result of this run cannot be read: {error.Message}";
        }
    }

    partial void OnFolderPathChanged(string value)
    {
        if (_index is { } index && !string.Equals(Path.TrimEndingDirectorySeparator(index.Folder), Path.TrimEndingDirectorySeparator(value), StringComparison.Ordinal))
        {
            SetIndex(null);
        }

        UpdateWatcher();
        StoredIndexLoad = LoadStoredIndexAsync(value);
        OnInputChanged();
    }

    private async Task LoadStoredIndexAsync(string folder)
    {
        if (folder.Length == 0 || !Directory.Exists(folder))
        {
            return;
        }

        string file = SearchIndexStore.PathFor(Services.Pages.DirectoryFor("search"), folder);
        var stored = await Task.Run(() => SearchIndexStore.Load(file)).ConfigureAwait(false);
        Services.Dispatcher.Post(() =>
        {
            if (stored is not null && FolderPath == folder && _index is null)
            {
                SetIndex(stored);
            }
        });
    }

    private void SetIndex(SearchIndex? index)
    {
        _index = index;
        IndexSummary = index is null ? "" : $"{index.FileCount} documents, {index.Chunks.Count} passages, {index.ModelId}";
        OnPropertyChanged(nameof(HasIndex));
        OnInputChanged();
    }

    partial void OnQueryChanged(string value) => OnInputChanged();

    private void OnInputChanged()
    {
        OnPropertyChanged(nameof(RunHint));
        IndexCommand.NotifyCanExecuteChanged();
        SearchCommand.NotifyCanExecuteChanged();
        AskCommand.NotifyCanExecuteChanged();
        OnPropertyChanged(nameof(CanAskModel));
    }

    [RelayCommand]
    private async Task ChooseFolderAsync()
    {
        if (await Services.Platform.PickFolderAsync("Choose a folder of documents") is { Length: > 0 } folder)
        {
            FolderPath = folder;
        }
    }

    private bool CanIndex() => HasModel && FolderPath.Length > 0 && Directory.Exists(FolderPath) && !Queue.IsBusy;

    /// <summary>Reads, cuts and embeds the documents of the folder; a stored index is updated, not rebuilt.</summary>
    [RelayCommand(CanExecute = nameof(CanIndex))]
    private void Index()
    {
        if (!CanIndex())
        {
            return;
        }

        var model = SelectedModel!;
        string folder = FolderPath;
        var options = Panel.Options;
        var values = Parameters.Snapshot();
        var previous = _index is { } current && string.Equals(current.Folder, folder, StringComparison.Ordinal) ? current : null;
        Parameters.Persist();
        Notice = "";
        Queue.Enqueue("Index " + (Path.GetFileName(folder.TrimEnd('/', '\\')) is { Length: > 0 } name ? name : folder), model.Id, context => IndexAsync(model, folder, previous, options, values, context));
    }

    private bool CanSearch() => HasModel && Query.Trim().Length > 0 && HasIndex && !Queue.IsBusy;

    /// <summary>Lists the passages closest to the question.</summary>
    [RelayCommand(CanExecute = nameof(CanSearch))]
    private void Search() => Enqueue(asker: null);

    private bool CanAsk() => CanSearch() && ChatModels.SelectedModel is not null;

    /// <summary>Lists the closest passages and has the chat model answer from them, citing the passages.</summary>
    [RelayCommand(CanExecute = nameof(CanAsk))]
    private void Ask() => Enqueue(asker: ChatModels.SelectedModel?.Model);

    private void Enqueue(ILoadedModel? asker)
    {
        if (!HasModel || _index is not { } index || Query.Trim().Length == 0 || Queue.IsBusy)
        {
            return;
        }

        var model = SelectedModel!;
        var reranker = Rerankers.SelectedModel?.Model;
        var options = Panel.Options;
        if (options.Mode != SearchMode.Keyword && index.ModelId != model.Id)
        {
            Notice = $"The index was made with {index.ModelId}. Index the folder again to search with {model.Id}.";
            return;
        }

        string query = Query.Trim();
        var values = Parameters.Snapshot();
        Parameters.Persist();
        Notice = "";
        Answer = "";
        AnswerCitations.Clear();
        string title = (asker is null ? "" : "Ask: ") + (query.Length > 60 ? query[..60] + "…" : query);
        Queue.Enqueue(title, model.Id, context => SearchAsync(model, options.Rerank ? reranker : null, asker, index, query, options, values, context));
    }

    /// <summary>Cancels the queued and running work.</summary>
    [RelayCommand]
    private void Cancel() => Queue.CancelAll();

    /// <summary>Opens the document of a result in the system's program, at its page or line where a program that can do that is installed.</summary>
    /// <param name="result">The result.</param>
    [RelayCommand]
    private void OpenResult(SearchResultViewModel result)
    {
        if (File.Exists(result.Hit.File))
        {
            Services.Platform.OpenAt(result.Hit.File, result.Hit.Page, result.Hit.Line);
        }
    }

    /// <summary>Copies a passage to the clipboard.</summary>
    /// <param name="result">The result.</param>
    [RelayCommand]
    private Task CopyResultAsync(SearchResultViewModel result) => Services.Platform.CopyTextAsync(result.Text);

    /// <summary>Marks the passage a citation of the answer points to.</summary>
    /// <param name="result">The cited result.</param>
    [RelayCommand]
    private void FocusResult(SearchResultViewModel result)
    {
        foreach (var other in Results)
        {
            other.IsFocused = ReferenceEquals(other, result);
        }
    }

    private async Task IndexAsync(ILoadedModel model, string folder, SearchIndex? previous, SearchOptions options, Dictionary<string, string> values, RunContext context)
    {
        var watch = Stopwatch.StartNew();
        RunOutcome outcome = RunOutcome.Completed;
        string? error = null;
        IndexUpdate? update = null;
        SearchIndex? latest = null;
        string file = SearchIndexStore.PathFor(Services.Pages.DirectoryFor("search"), folder);
        try
        {
            context.Report(null, "Loading the model");
            using var lease = await model.AcquireEmbedderAsync(context.CancellationToken).ConfigureAwait(false);
            update = await _indexer.UpdateAsync(folder, previous, model.Id, lease.Embedder, DocumentPrompt(lease.Embedder), options.ChunkCharacters, options.OverlapCharacters,
                context.Report, checkpoint: index =>
                {
                    SearchIndexStore.Save(file, index);
                    latest = index;
                }, context.CancellationToken).ConfigureAwait(false);
            SearchIndexStore.Save(file, update.Index);
            latest = update.Index;
        }
        catch (OperationCanceledException)
        {
            outcome = RunOutcome.Cancelled;
        }
        catch (Exception failure) when (failure is not OutOfMemoryException)
        {
            outcome = RunOutcome.Failed;
            error = failure.GetBaseException().Message;
        }

        double seconds = watch.Elapsed.TotalSeconds;
        var metadata = new Dictionary<string, string>();
        if (update is not null)
        {
            var number = CultureInfo.InvariantCulture;
            metadata["documents"] = update.Index.FileCount.ToString(number);
            metadata["passages"] = update.Index.Chunks.Count.ToString(number);
            metadata["time"] = $"{seconds:0.0} s";
            metadata["added"] = update.Added.ToString(number);
            metadata["changed"] = update.Changed.ToString(number);
            metadata["removed"] = update.Removed.ToString(number);
            if (update.Failed > 0)
            {
                metadata["unreadable"] = update.Failed.ToString(number);
            }
        }

        if (latest is not null)
        {
            var shown = latest;
            Services.Dispatcher.Post(() =>
            {
                SetIndex(shown);
                if (outcome == RunOutcome.Completed)
                {
                    Results.Clear();
                    ResultSummary = "";
                    OnPropertyChanged(nameof(HasResults));
                }
            });
        }

        var record = new RunRecord
        {
            Id = PageArchive.NewId(),
            Title = "Index " + Path.GetFileName(folder.TrimEnd('/', '\\')),
            Model = model.Id,
            Outcome = outcome,
            Seconds = seconds,
            Error = error,
            Input = new Dictionary<string, string> { ["folder"] = folder },
            Options = values,
            Metadata = metadata,
        };
        if (outcome != RunOutcome.Cancelled)
        {
            Archived(record);
        }

        Finish(outcome, error, context);
    }

    private async Task SearchAsync(ILoadedModel model, ILoadedModel? reranker, ILoadedModel? asker, SearchIndex index, string query, SearchOptions options, Dictionary<string, string> values, RunContext context)
    {
        var watch = Stopwatch.StartNew();
        RunOutcome outcome = RunOutcome.Completed;
        string? error = null;
        string answer = "";
        List<SearchHitRecord> hits = [];
        try
        {
            hits = await RetrieveAsync(model, reranker, index, query, options, context).ConfigureAwait(false);
            if (asker is not null && hits.Count > 0)
            {
                var shown = hits;
                Services.Dispatcher.Post(() => Show(shown, index.Folder, query));
                answer = await AnswerAsync(asker, query, hits, context).ConfigureAwait(false);
            }
        }
        catch (OperationCanceledException)
        {
            outcome = RunOutcome.Cancelled;
        }
        catch (Exception failure) when (failure is not OutOfMemoryException)
        {
            outcome = RunOutcome.Failed;
            error = failure.GetBaseException().Message;
        }

        double seconds = watch.Elapsed.TotalSeconds;
        var metadata = new Dictionary<string, string>
        {
            ["passages"] = hits.Count.ToString(CultureInfo.InvariantCulture),
            ["time"] = $"{seconds * 1000:0} ms",
        };
        if (options.Mode != SearchMode.Semantic)
        {
            metadata["mode"] = options.Mode;
        }

        if (reranker is not null)
        {
            metadata["reranker"] = reranker.Id;
        }

        if (asker is not null)
        {
            metadata["answered by"] = asker.Id;
        }

        if (outcome == RunOutcome.Completed)
        {
            Services.Dispatcher.Post(() =>
            {
                Show(hits, index.Folder, query);
                ShowAnswer(answer);
                ResultSummary = string.Join(" · ", metadata.Select(pair => $"{pair.Key}: {pair.Value}"));
            });
        }

        var file = new SearchResultsFile { Query = query, Folder = index.Folder, Model = model.Id, Mode = options.Mode, Answer = answer, Hits = hits };
        var record = new RunRecord
        {
            Id = PageArchive.NewId(),
            Title = (asker is null ? "" : "Ask: ") + query,
            Model = model.Id,
            Outcome = outcome,
            Seconds = seconds,
            Error = error,
            Input = new Dictionary<string, string> { ["query"] = query, ["folder"] = index.Folder },
            Options = values,
            Metadata = metadata,
            Preview = hits.Count > 0 ? Path.GetFileName(hits[0].File) + ": " + (hits[0].Text.Length > 160 ? hits[0].Text[..160] + "…" : hits[0].Text) : null,
            Files = outcome == RunOutcome.Completed ? [ResultsName] : [],
        };
        var files = outcome != RunOutcome.Completed ? null : new Dictionary<string, byte[]> { [ResultsName] = JsonSerializer.SerializeToUtf8Bytes(file, StudioJsonContext.Default.SearchResultsFile) };
        if (outcome != RunOutcome.Cancelled)
        {
            Archived(record, files);
        }

        Finish(outcome, error, context);
    }

    private async Task<List<SearchHitRecord>> RetrieveAsync(ILoadedModel model, ILoadedModel? reranker, SearchIndex index, string query, SearchOptions options, RunContext context)
    {
        int top = Math.Max(1, options.TopK);
        int wanted = reranker is null ? top : Math.Max(top * 4, 20);
        int pool = Math.Max(wanted, CandidatePool);
        IReadOnlyList<(int Index, double Score)> ranked;
        if (options.Mode == SearchMode.Keyword)
        {
            context.Report(0.5, "Matching keywords");
            var keyword = await Task.Run(() => index.Keyword(query, wanted), context.CancellationToken).ConfigureAwait(false);
            double best = keyword.Count > 0 ? keyword[0].Score : 1;
            ranked = [.. keyword.Select(hit => (hit.Index, hit.Score / best))];
        }
        else
        {
            context.Report(null, "Loading the model");
            using var lease = await model.AcquireEmbedderAsync(context.CancellationToken).ConfigureAwait(false);
            context.Report(0.5, "Embedding the question");
            var embedded = await Task.Run(() => lease.Embedder.Embed([query], new TextEmbeddingOptions { Prompt = QueryPrompt(lease.Embedder) }), context.CancellationToken).ConfigureAwait(false);
            if (options.Mode == SearchMode.Hybrid)
            {
                ranked = await Task.Run(() => Fuse(index, embedded.Vectors[0], query, pool, wanted), context.CancellationToken).ConfigureAwait(false);
            }
            else
            {
                ranked = await Task.Run(() => index.Rank(embedded.Vectors[0], wanted).Select(hit => (hit.Index, (double)hit.Score)).ToList(), context.CancellationToken).ConfigureAwait(false);
            }
        }

        var hits = ranked.Select(hit => ToHit(index.Chunks[hit.Index], hit.Score)).ToList();
        if (reranker is not null && hits.Count > 0)
        {
            context.Report(0.75, $"Reranking {hits.Count} passages");
            using var rerankLease = await reranker.AcquireRerankerAsync(context.CancellationToken).ConfigureAwait(false);
            var scored = await Task.Run(() => rerankLease.Reranker.Score(query, [.. hits.Select(hit => hit.Text)]), context.CancellationToken).ConfigureAwait(false);
            hits = [.. hits.Select((hit, i) => hit with { Score = scored.Scores[i] }).OrderByDescending(hit => hit.Score).Take(top)];
        }

        return hits;
    }

    private static IReadOnlyList<(int Index, double Score)> Fuse(SearchIndex index, float[] question, string query, int pool, int wanted)
    {
        var semantic = index.Rank(question, pool).Select(hit => hit.Index).ToList();
        var keyword = index.Keyword(query, pool).Select(hit => hit.Index).ToList();
        double best = 2.0 / (RankFusion.DefaultK + 1);
        return [.. RankFusion.Fuse([semantic, keyword]).Take(wanted).Select(hit => (hit.Id, hit.Score / best))];
    }

    private static SearchHitRecord ToHit(IndexedChunk chunk, double score) => new()
    {
        File = chunk.File,
        Start = chunk.Start,
        Text = chunk.Text,
        Score = score,
        Page = chunk.Page,
        Line = chunk.Line,
        Heading = chunk.Heading,
    };

    private async Task<string> AnswerAsync(ILoadedModel asker, string query, List<SearchHitRecord> hits, RunContext context)
    {
        context.Report(0.85, $"Asking {asker.Id}");
        var passages = hits.Select(hit => (Label: Path.GetFileName(hit.File) + (LocationLabel.For(hit.File, hit.Page, hit.Line, hit.Heading) is { Length: > 0 } place ? ", " + place : ""), hit.Text)).ToList();
        var (input, _) = AnswerPrompt.Build(query, passages, asker.ContextLength, asker.CountTokens);
        var text = new System.Text.StringBuilder();
        var clock = Stopwatch.StartNew();
        long lastShown = -1000;
        await foreach (var update in asker.StreamChatAsync(input, context.CancellationToken).ConfigureAwait(false))
        {
            if (update.Kind == ChatStreamKind.Content)
            {
                text.Append(update.Text);
                if (clock.ElapsedMilliseconds - lastShown >= 100)
                {
                    lastShown = clock.ElapsedMilliseconds;
                    string partial = text.ToString();
                    Services.Dispatcher.Post(() => Answer = partial);
                }
            }
            else if (update.Kind == ChatStreamKind.Finished && update.Error is { } failure)
            {
                throw new InvalidOperationException($"{asker.Id} could not answer: {failure}");
            }
        }

        return text.ToString().Trim();
    }

    private void Finish(RunOutcome outcome, string? error, RunContext context)
    {
        if (outcome == RunOutcome.Cancelled)
        {
            throw new OperationCanceledException(context.CancellationToken);
        }

        if (error is not null)
        {
            Services.Dispatcher.Post(() => Notice = error);
            throw new InvalidOperationException(error);
        }
    }

    private void Show(IEnumerable<SearchHitRecord> hits, string folder, string query)
    {
        Results.Clear();
        int number = 0;
        foreach (var hit in hits)
        {
            Results.Add(new SearchResultViewModel(hit, folder, ++number, query));
        }

        OnPropertyChanged(nameof(HasResults));
    }

    private void ShowAnswer(string answer)
    {
        Answer = answer;
        AnswerCitations.Clear();
        foreach (int number in AnswerPrompt.Citations(answer, Results.Count))
        {
            AnswerCitations.Add(Results[number - 1]);
        }
    }

    private static string? QueryPrompt(ITextEmbedder embedder) => embedder.Prompts.ContainsKey("query") ? "query" : null;

    private static string? DocumentPrompt(ITextEmbedder embedder) => embedder.Prompts.ContainsKey("document") ? "document" : null;

    private static ParameterPanelViewModel<SearchOptions> CreateParameters(StudioServices services)
    {
        ParameterViewModel[] parameters =
        [
            new ChoiceParameterViewModel("mode", "Search mode", "Semantic ranks by meaning (embeddings), Keyword by the words themselves (BM25), Hybrid combines both rankings by reciprocal rank fusion, which finds both paraphrases and exact names, numbers and codes.",
                [new(SearchMode.Semantic, "Semantic"), new(SearchMode.Keyword, "Keyword (BM25)"), new(SearchMode.Hybrid, "Hybrid")]),
            new NumberParameterViewModel("topk", "Results", "How many passages a search lists.", 1, 50, 1, 5),
            new NumberParameterViewModel("chunk", "Passage length", "Documents are cut into passages of about this many characters, along headings, pages and slides first; shorter passages find details, longer ones keep context. Changing it indexes the folder again.", 200, 4000, 100, 1000),
            new NumberParameterViewModel("overlap", "Overlap", "Characters that consecutive passages share, so an answer on a boundary is not lost.", 0, 1000, 50, 150),
            new BoolParameterViewModel("rerank", "Rerank results", "Score the closest passages again with the chosen reranking model (a cross-encoder reads the question and each passage together) and list the best of them. Shown when a reranking model is loaded."),
        ];
        ParameterPreset[] presets =
        [
            new("Precise", new() { ["chunk"] = "500", ["overlap"] = "100", ["topk"] = "8" }, BuiltIn: true),
            new("Balanced", new() { ["chunk"] = "1000", ["overlap"] = "150", ["topk"] = "5" }, BuiltIn: true),
            new("Context", new() { ["chunk"] = "2000", ["overlap"] = "300", ["topk"] = "4" }, BuiltIn: true),
        ];
        return new ParameterPanelViewModel<SearchOptions>(services.Pages, "search", parameters,
            panel => new SearchOptions(panel.GetInt("topk"), panel.GetInt("chunk"), panel.GetInt("overlap"), panel.GetBool("rerank"), panel.GetChoice("mode")), presets);
    }
}
