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
using Sharplight.Studio.ViewModels.Inference;
using Sharplight.Studio.ViewModels.Inference.Parameters;

namespace Sharplight.Studio.ViewModels.Search;

/// <summary>
/// The Search page (kind <c>embeddings</c>): a folder of documents is read, cut into passages and embedded by the chosen text
/// embedding model (the index lives in memory for that model), and a question is embedded the same way and answered with the closest
/// passages. Built from the shared inference blocks (picker, parameters, queue, history, outputs); both indexing and searching are runs
/// of the queue, so the embedding model's work never touches the UI thread and can be cancelled.
/// </summary>
internal sealed partial class SearchPageViewModel : InferencePageViewModel
{
    private const string ResultsName = "results.json";
    private const int MaxFiles = 5000;
    private const int BatchSize = 16;

    private SearchIndex? _index;

    /// <param name="services">The services.</param>
    /// <param name="navigator">Moves to the library and Discover.</param>
    public SearchPageViewModel(StudioServices services, INavigator navigator)
        : base(services, navigator, ModelKindCatalog.Embeddings, "search", CreateParameters(services))
    {
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
        return base.ActivateAsync();
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

            Show(file.Hits, file.Folder);
            ResultSummary = string.Join(" · ", record.Metadata.Select(pair => $"{pair.Key}: {pair.Value}"));
            Parameters.Apply(record.Options);
        }
        catch (Exception error) when (error is IOException or JsonException or UnauthorizedAccessException)
        {
            Notice = $"The result of this run cannot be read: {error.Message}";
        }
    }

    partial void OnFolderPathChanged(string value) => OnInputChanged();

    partial void OnQueryChanged(string value) => OnInputChanged();

    private void OnInputChanged()
    {
        OnPropertyChanged(nameof(RunHint));
        IndexCommand.NotifyCanExecuteChanged();
        SearchCommand.NotifyCanExecuteChanged();
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

    /// <summary>Reads, cuts and embeds the documents of the folder.</summary>
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
        Parameters.Persist();
        Notice = "";
        Queue.Enqueue("Index " + (Path.GetFileName(folder.TrimEnd('/', '\\')) is { Length: > 0 } name ? name : folder), model.Id, context => IndexAsync(model, folder, options, values, context));
    }

    private bool CanSearch() => HasModel && Query.Trim().Length > 0 && HasIndex && !Queue.IsBusy;

    /// <summary>Embeds the question and lists the closest passages.</summary>
    [RelayCommand(CanExecute = nameof(CanSearch))]
    private void Search()
    {
        if (!HasModel || _index is not { } index)
        {
            return;
        }

        var model = SelectedModel!;
        var reranker = Rerankers.SelectedModel?.Model;
        if (index.ModelId != model.Id)
        {
            Notice = $"The index was made with {index.ModelId}. Index the folder again to search with {model.Id}.";
            return;
        }

        string query = Query.Trim();
        var options = Panel.Options;
        var values = Parameters.Snapshot();
        Parameters.Persist();
        Notice = "";
        Queue.Enqueue(query.Length > 60 ? query[..60] + "…" : query, model.Id, context => SearchAsync(model, options.Rerank ? reranker : null, index, query, options, values, context));
    }

    /// <summary>Cancels the queued and running work.</summary>
    [RelayCommand]
    private void Cancel() => Queue.CancelAll();

    /// <summary>Opens the document of a result in the system's program.</summary>
    /// <param name="result">The result.</param>
    [RelayCommand]
    private void OpenResult(SearchResultViewModel result)
    {
        if (File.Exists(result.Hit.File))
        {
            Services.Platform.Open(result.Hit.File);
        }
    }

    /// <summary>Copies a passage to the clipboard.</summary>
    /// <param name="result">The result.</param>
    [RelayCommand]
    private Task CopyResultAsync(SearchResultViewModel result) => Services.Platform.CopyTextAsync(result.Text);

    private async Task IndexAsync(ILoadedModel model, string folder, SearchOptions options, Dictionary<string, string> values, RunContext context)
    {
        var watch = Stopwatch.StartNew();
        RunOutcome outcome = RunOutcome.Completed;
        string? error = null;
        SearchIndex? built = null;
        int skipped = 0;
        try
        {
            context.Report(null, "Listing documents");
            var files = Directory.EnumerateFiles(folder, "*", new EnumerationOptions { RecurseSubdirectories = true, IgnoreInaccessible = true })
                .Where(Services.Documents.CanExtract).Order(StringComparer.Ordinal).Take(MaxFiles).ToList();
            context.Report(null, "Loading the model");
            using var lease = await model.AcquireEmbedderAsync(context.CancellationToken).ConfigureAwait(false);
            var chunks = new List<IndexedChunk>();
            for (int i = 0; i < files.Count; i++)
            {
                context.CancellationToken.ThrowIfCancellationRequested();
                context.Report((double)i / Math.Max(1, files.Count), $"{i + 1} of {files.Count} documents");
                string text;
                try
                {
                    text = (await Services.Documents.ExtractAsync(files[i], context.CancellationToken).ConfigureAwait(false)).Text;
                }
                catch (Exception failure) when (failure is IOException or NotSupportedException or UnauthorizedAccessException or InvalidDataException)
                {
                    skipped++;
                    continue;
                }

                var passages = TextChunker.Split(text, options.ChunkCharacters, options.OverlapCharacters);
                for (int start = 0; start < passages.Count; start += BatchSize)
                {
                    var batch = passages.Skip(start).Take(BatchSize).ToList();
                    var vectors = await Task.Run(() => lease.Embedder.Embed([.. batch.Select(passage => passage.Text)], new EmbedOptions { Prompt = DocumentPrompt(lease.Embedder) }),
                        context.CancellationToken).ConfigureAwait(false);
                    for (int j = 0; j < batch.Count; j++)
                    {
                        chunks.Add(new IndexedChunk(files[i], batch[j].Start, batch[j].Text, SearchIndex.Normalize(vectors.Vectors[j])));
                    }
                }
            }

            built = new SearchIndex(folder, model.Id, files.Count - skipped, chunks);
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
        if (built is not null)
        {
            metadata["documents"] = built.FileCount.ToString(CultureInfo.InvariantCulture);
            metadata["passages"] = built.Chunks.Count.ToString(CultureInfo.InvariantCulture);
            metadata["time"] = $"{seconds:0.0} s";
            if (skipped > 0)
            {
                metadata["unreadable"] = skipped.ToString(CultureInfo.InvariantCulture);
            }

            Services.Dispatcher.Post(() =>
            {
                _index = built;
                Results.Clear();
                ResultSummary = "";
                IndexSummary = $"{built.FileCount} documents, {built.Chunks.Count} passages, {built.ModelId}";
                OnPropertyChanged(nameof(HasIndex));
                OnPropertyChanged(nameof(HasResults));
                OnInputChanged();
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

    private async Task SearchAsync(ILoadedModel model, ILoadedModel? reranker, SearchIndex index, string query, SearchOptions options, Dictionary<string, string> values, RunContext context)
    {
        var watch = Stopwatch.StartNew();
        RunOutcome outcome = RunOutcome.Completed;
        string? error = null;
        List<SearchHitRecord> hits = [];
        try
        {
            context.Report(null, "Loading the model");
            using var lease = await model.AcquireEmbedderAsync(context.CancellationToken).ConfigureAwait(false);
            context.Report(0.5, "Embedding the question");
            var embedded = await Task.Run(() => lease.Embedder.Embed([query], new EmbedOptions { Prompt = QueryPrompt(lease.Embedder) }), context.CancellationToken).ConfigureAwait(false);
            int top = Math.Max(1, options.TopK);
            var nearest = await Task.Run(() => index.Nearest(embedded.Vectors[0], reranker is null ? top : Math.Max(top * 4, 20)), context.CancellationToken).ConfigureAwait(false);
            hits = [.. nearest.Select(hit => new SearchHitRecord { File = hit.Chunk.File, Start = hit.Chunk.Start, Text = hit.Chunk.Text, Score = hit.Score })];
            if (reranker is not null && hits.Count > 0)
            {
                context.Report(0.75, $"Reranking {hits.Count} passages");
                using var rerankLease = await reranker.AcquireRerankerAsync(context.CancellationToken).ConfigureAwait(false);
                var scored = await Task.Run(() => rerankLease.Reranker.Score(query, [.. hits.Select(hit => hit.Text)]), context.CancellationToken).ConfigureAwait(false);
                hits = [.. hits.Select((hit, i) => hit with { Score = scored.Scores[i] }).OrderByDescending(hit => hit.Score).Take(top)];
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
        if (reranker is not null)
        {
            metadata["reranker"] = reranker.Id;
        }

        if (outcome == RunOutcome.Completed)
        {
            Services.Dispatcher.Post(() =>
            {
                Show(hits, index.Folder);
                ResultSummary = string.Join(" · ", metadata.Select(pair => $"{pair.Key}: {pair.Value}"));
            });
        }

        var file = new SearchResultsFile { Query = query, Folder = index.Folder, Model = model.Id, Hits = hits };
        var record = new RunRecord
        {
            Id = PageArchive.NewId(),
            Title = query,
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

    private void Show(IEnumerable<SearchHitRecord> hits, string folder)
    {
        Results.Clear();
        foreach (var hit in hits)
        {
            Results.Add(new SearchResultViewModel(hit, folder));
        }

        OnPropertyChanged(nameof(HasResults));
    }

    private static string? QueryPrompt(ITextEmbedder embedder) => embedder.Prompts.ContainsKey("query") ? "query" : null;

    private static string? DocumentPrompt(ITextEmbedder embedder) => embedder.Prompts.ContainsKey("document") ? "document" : null;

    private static ParameterPanelViewModel<SearchOptions> CreateParameters(StudioServices services)
    {
        ParameterViewModel[] parameters =
        [
            new NumberParameterViewModel("topk", "Results", "How many passages a search lists.", 1, 50, 1, 5),
            new NumberParameterViewModel("chunk", "Passage length", "Documents are cut into passages of about this many characters; shorter passages find details, longer ones keep context.", 200, 4000, 100, 1000),
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
            panel => new SearchOptions(panel.GetInt("topk"), panel.GetInt("chunk"), panel.GetInt("overlap"), panel.GetBool("rerank")), presets);
    }
}
