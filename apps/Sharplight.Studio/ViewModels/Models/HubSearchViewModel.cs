using System.Collections.ObjectModel;
using System.ComponentModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Hub;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services;
using Sharplight.Studio.Services.Compatibility;
using Sharplight.Studio.Services.Fit;
using Sharplight.Studio.Services.Hub;
using Sharplight.Studio.Services.Library;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>Searching the Hugging Face Hub and choosing a variant to download.</summary>
internal sealed partial class HubSearchViewModel : ObservableObject
{
    private const int ResultLimit = 60;

    private readonly StudioServices _services;
    private readonly DownloadsViewModel _downloads;
    private readonly Func<IReadOnlySet<string>> _storedNames;
    private readonly FitAdvisor _advisor;
    private readonly List<HubResultViewModel> _allResults = [];
    private readonly List<VariantViewModel> _allVariants = [];
    private CancellationTokenSource? _details;
    private RepositoryDetails? _openDetails;
    private Task? _highlightsLoad;
    private bool _highlightsFailed;
    private ModelTask? _highlightsTask;
    private IReadOnlyList<HighlightSection> _rawHighlights = [];
    private DiscoverFilterSettings _searched;
    private long? _capacity;

    /// <summary>The user picked a mode; the mode no longer follows the model type.</summary>
    private bool _modeChosen;

    public HubSearchViewModel(StudioServices services, DownloadsViewModel downloads, Func<IReadOnlySet<string>> storedNames,
        Func<IReadOnlyList<ModelManifest>>? storedModels = null)
    {
        _services = services;
        _downloads = downloads;
        _storedNames = storedNames;
        _advisor = new FitAdvisor(services.Runtime, services.Settings);
        Filters = new DiscoverFiltersViewModel(services.Settings);
        _searched = Filters.Current;
        ShowOnlySupported = services.Settings.Current.ShowOnlySupported;
        Filters.FiltersChanged += (_, _) => _ = OnFiltersChangedAsync();
        Verified = new VerifiedCatalogViewModel(services, downloads, storedModels ?? (() => []));
        Feed = new FeedCatalogViewModel(services, storedModels ?? (() => []));
        Verified.Loaded += (_, _) => ChooseMode();
    }

    /// <summary>The verified-model catalog, Discover's default list when it has models of the selected type.</summary>
    public VerifiedCatalogViewModel Verified { get; }

    /// <summary>The Sharplight model feed: signed, preconverted <c>.slm</c> packages.</summary>
    public FeedCatalogViewModel Feed { get; }

    /// <summary>Discover shows the verified catalog.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsHubMode))]
    public partial bool IsVerifiedMode { get; set; }

    /// <summary>Discover shows the Sharplight model feed.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsHubMode))]
    public partial bool IsFeedMode { get; set; }

    /// <summary>Discover shows the Hub search.</summary>
    public bool IsHubMode => !IsVerifiedMode && !IsFeedMode;

    [RelayCommand]
    private void ShowVerified()
    {
        _modeChosen = true;
        IsFeedMode = false;
        IsVerifiedMode = true;
    }

    [RelayCommand]
    private void ShowHubSearch()
    {
        _modeChosen = true;
        IsFeedMode = false;
        IsVerifiedMode = false;
    }

    [RelayCommand]
    private void ShowFeed()
    {
        _modeChosen = true;
        IsVerifiedMode = false;
        IsFeedMode = true;
        _ = Feed.LoadAsync();
    }

    /// <summary>Follows the model type: the verified list when the catalog has models of it, else the Hub search (until a mode is picked).</summary>
    private void ChooseMode()
    {
        var task = Filters.Current.Task;
        Verified.SelectTask(task);
        if (!_modeChosen)
        {
            IsVerifiedMode = Verified.HasEntriesFor(task);
        }
    }

    /// <summary>The filters and sort order (model type, format, quantization, size, license, gated).</summary>
    public DiscoverFiltersViewModel Filters { get; }

    /// <summary>Search text.</summary>
    [ObservableProperty]
    public partial string Query { get; set; } = "";

    /// <summary>
    /// List only repositories and quantizations whose badge is <c>Supported</c> (saved in the settings). Quantizations whose metadata is
    /// still being read stay listed until their badge is known.
    /// </summary>
    [ObservableProperty]
    public partial bool ShowOnlySupported { get; set; }

    /// <summary>The search results shown (all of them, or the supported ones with <see cref="ShowOnlySupported"/>).</summary>
    public ObservableCollection<HubResultViewModel> Results { get; } = [];

    /// <summary>How many search results <see cref="ShowOnlySupported"/> and the client-side filters hide, as a sentence; empty when none.</summary>
    [ObservableProperty]
    public partial string HiddenResultsText { get; set; } = "";

    /// <summary>The highlighted models shown before anything is searched.</summary>
    public ObservableCollection<HighlightSectionViewModel> Highlights { get; } = [];

    /// <summary>The highlights are shown instead of search results (nothing is searched).</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsShowingResults))]
    public partial bool IsShowingHighlights { get; set; } = true;

    /// <summary>Search results are shown.</summary>
    public bool IsShowingResults => !IsShowingHighlights;

    /// <summary>The model type the highlights are of (everything the engine runs, when no type is chosen), as a heading.</summary>
    [ObservableProperty]
    public partial string HighlightsHeading { get; set; } = "Language models Sharplight supports";

    /// <summary>The highlights are being fetched.</summary>
    [ObservableProperty]
    public partial bool IsLoadingHighlights { get; set; }

    /// <summary>Why no highlights are shown (the Hub could not be reached), or empty.</summary>
    [ObservableProperty]
    public partial string HighlightsStatus { get; set; } = "";

    /// <summary>The result whose details are shown.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasSelection), nameof(SelectedId))]
    public partial HubResultViewModel? SelectedResult { get; set; }

    /// <summary>Variants of the selected repository that are shown (see <see cref="ShowOnlySupported"/>).</summary>
    public ObservableCollection<VariantViewModel> Variants { get; } = [];

    /// <summary>How many quantizations <see cref="ShowOnlySupported"/> hides, as a sentence; empty when none.</summary>
    [ObservableProperty]
    public partial string HiddenVariantsText { get; set; } = "";

    /// <summary>The variant to download.</summary>
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(DownloadCommand))]
    [NotifyPropertyChangedFor(nameof(SelectedFit), nameof(HasFitWarning), nameof(HasFitOk), nameof(SelectedCompatibility))]
    public partial VariantViewModel? SelectedVariant { get; set; }

    /// <summary>The likely compatibility of the selected variant (shown above the download button; downloading stays possible).</summary>
    public CompatibilityBadgeViewModel? SelectedCompatibility => SelectedVariant?.Compatibility;

    /// <summary>The latest read of a selected variant's weight formats (for tests).</summary>
    internal Task CompatibilityCheck { get; private set; } = Task.CompletedTask;

    /// <summary>The approximate fit of the selected variant, once estimated.</summary>
    public FitBadgeViewModel? SelectedFit => SelectedVariant?.Fit;

    /// <summary>The selected variant would not load as is (shown above the download button; downloading stays possible).</summary>
    public bool HasFitWarning => SelectedFit is { IsGood: false };

    /// <summary>The selected variant fits.</summary>
    public bool HasFitOk => SelectedFit is { IsGood: true };

    /// <summary>The variants' fit is being estimated.</summary>
    [ObservableProperty]
    public partial bool IsEstimatingFit { get; set; }

    /// <summary>The latest fit estimate of the variants (for tests).</summary>
    internal Task FitEstimate { get; private set; } = Task.CompletedTask;

    /// <summary>Files of the selected repository, as <c>path (size)</c>.</summary>
    public ObservableCollection<string> Files { get; } = [];

    /// <summary>A search is running.</summary>
    [ObservableProperty]
    public partial bool IsSearching { get; set; }

    /// <summary>Details are loading.</summary>
    [ObservableProperty]
    public partial bool IsLoadingDetails { get; set; }

    /// <summary>Status or error text.</summary>
    [ObservableProperty]
    public partial string Status { get; set; } = DefaultStatus;

    private const string DefaultStatus = "Search Hugging Face, or pick a model type and filters to browse.";

    /// <summary>Format the selected repository is pulled as.</summary>
    [ObservableProperty]
    public partial string? DetailFormat { get; set; }

    /// <summary>The parts of the selected multi-component repository (a Diffusers pipeline, whisper.cpp files), shown as one model.</summary>
    public ObservableCollection<ComponentViewModel> Components { get; } = [];

    /// <summary>The selected repository is made of several components.</summary>
    [ObservableProperty]
    public partial bool IsMultiComponent { get; set; }

    /// <summary>The multi-component repository is downloaded as one model (its variants, such as the weight precision, are listed below its parts).</summary>
    [ObservableProperty]
    public partial bool IsPulledAsOne { get; set; }

    /// <summary>Why the selected repository cannot be downloaded (its kind or layout is not supported yet), or empty.</summary>
    [ObservableProperty]
    public partial string NoDownloadReason { get; set; } = "";

    /// <summary>What to do about a gated repository (accept its terms, set a token), or empty for an open one.</summary>
    [ObservableProperty]
    public partial string GatedNotice { get; set; } = "";

    /// <summary>A result is selected.</summary>
    public bool HasSelection => SelectedResult is not null;

    /// <summary>The selected repository id (empty without a selection).</summary>
    public string SelectedId => SelectedResult?.Id ?? "";

    partial void OnSelectedResultChanged(HubResultViewModel? oldValue, HubResultViewModel? newValue)
    {
        if (oldValue is not null)
        {
            oldValue.IsSelected = false;
        }

        if (newValue is not null)
        {
            newValue.IsSelected = true;
            _ = LoadDetailsAsync(newValue);
        }
    }

    partial void OnShowOnlySupportedChanged(bool value)
    {
        if (_services.Settings.Current.ShowOnlySupported != value)
        {
            _services.Settings.Update(settings => settings with { ShowOnlySupported = value });
        }

        ApplyResultFilter();
        ApplyVariantFilter();
        if (Filters.Current.Task is not null)
        {
            RenderHighlights();
        }
    }

    partial void OnQueryChanged(string value)
    {
        if (value.Trim().Length == 0 && !IsShowingHighlights && IsHighlightMode)
        {
            IsShowingHighlights = true;
            SelectedResult = null;
            Status = DefaultStatus;
        }
    }

    /// <summary>
    /// With no search text, the highlights of the chosen model type are shown (the default ones without a type), as long as no other
    /// filter is set and the sort order is the default; otherwise the Hub's listing in the chosen order is shown instead.
    /// </summary>
    private bool IsHighlightMode => Query.Trim().Length == 0 && Filters.Current with { Task = null } == new DiscoverFilterSettings();

    /// <summary>Reacts to a filter change: asks the Hub again when the query changed, otherwise only filters the results shown.</summary>
    private async Task OnFiltersChangedAsync()
    {
        var current = Filters.Current;
        if (current.Task != _searched.Task)
        {
            ChooseMode();
        }

        bool hubChanged = current.Task != _searched.Task || current.Format != _searched.Format || current.License != _searched.License || current.Sort != _searched.Sort;
        _searched = current;
        if (IsHighlightMode)
        {
            IsShowingHighlights = true;
            SelectedResult = null;
            Status = DefaultStatus;
            if (hubChanged)
            {
                _highlightsLoad = null;
                await LoadHighlightsAsync();
            }

            return;
        }

        if (!hubChanged && !IsShowingHighlights)
        {
            if (current.FitsThisMachine)
            {
                await UpdateCapacityAsync();
            }

            ApplyResultFilter();
            return;
        }

        await SearchAsync();
    }

    private async Task UpdateCapacityAsync()
    {
        try
        {
            var memory = await _advisor.GetMemoryAsync(CancellationToken.None);
            _capacity = DiscoverFilter.Capacity(memory.Where(entry => entry.IsGpu).Sum(entry => entry.UsableBytes),
                memory.Where(entry => !entry.IsGpu).Select(entry => entry.UsableBytes).DefaultIfEmpty(0).Max());
        }
        catch (Exception error) when (error is not OutOfMemoryException)
        {
            _capacity = null;
        }
    }

    /// <summary>
    /// Fetches the highlighted models once (the catalog caches them for a few hours, so opening Discover again does not ask the Hub).
    /// </summary>
    public Task LoadHighlightsAsync()
    {
        _ = Verified.LoadAsync();
        _ = Feed.LoadAsync();
        if (_highlightsLoad is null || _highlightsFailed || _highlightsTask != Filters.Current.Task)
        {
            _highlightsFailed = false;
            _highlightsLoad = FetchHighlightsAsync();
        }

        return _highlightsLoad;
    }

    /// <summary>Shows a repository's details (the highlight cards).</summary>
    [RelayCommand]
    private void SelectResult(HubResultViewModel? result)
    {
        if (result is not null)
        {
            SelectedResult = result;
        }
    }

    private async Task FetchHighlightsAsync()
    {
        var task = Filters.Current.Task;
        _highlightsTask = task;
        IsLoadingHighlights = true;
        HighlightsStatus = "";
        HighlightsHeading = task is { } chosen ? $"{ModelTaskTitles.Title(chosen)} models" : "Language models Sharplight supports";
        try
        {
            var sections = await _services.Hub.GetHighlightsAsync(task, CancellationToken.None);
            if (task != Filters.Current.Task)
            {
                return;
            }

            _rawHighlights = sections;
            RenderHighlights();
        }
        catch (Exception error) when (error is HubException or HttpRequestException or TaskCanceledException)
        {
            HighlightsStatus = "Hugging Face could not be reached, so no highlighted models are shown. Search to try again.";
            _highlightsFailed = true;
        }
        finally
        {
            IsLoadingHighlights = false;
        }
    }

    /// <summary>
    /// Shows the highlight sections. A model type's highlights carry every model of the type, so <see cref="ShowOnlySupported"/>
    /// drops those Sharplight cannot run (all of them, for a type the engine has no client for).
    /// </summary>
    private void RenderHighlights()
    {
        var task = Filters.Current.Task;
        Highlights.Clear();
        int total = 0;
        foreach (var section in _rawHighlights)
        {
            var items = section.Models.Select(model => new HubResultViewModel(model)).Where(item => task is null || !ShowOnlySupported || item.Compatibility.IsGood).ToList();
            total += section.Models.Count;
            if (items.Count > 0)
            {
                Highlights.Add(new HighlightSectionViewModel(section.Kind, items));
            }
        }

        if (Highlights.Count == 0)
        {
            HighlightsStatus = total > 0 && task is { } hidden
                ? $"Sharplight cannot run {ModelTaskTitles.Title(hidden).ToLowerInvariant()} models yet. Turn off Supported only to see what is on Hugging Face."
                : "No highlighted models right now. Search to find models.";
        }
        else
        {
            HighlightsStatus = "";
        }
    }

    /// <summary>
    /// Shows the results that pass <see cref="ShowOnlySupported"/> and the filters the Hub cannot apply (<see cref="DiscoverFilter"/>),
    /// in their order; a hidden selection moves to the first shown.
    /// </summary>
    private void ApplyResultFilter()
    {
        var filters = Filters.Current;
        var shown = _allResults.Where(IsShown).ToList();
        var selected = SelectedResult;
        if (!shown.SequenceEqual(Results))
        {
            Results.Clear();
            foreach (var result in shown)
            {
                Results.Add(result);
            }
        }

        // Clearing the list can clear the selection through the list's binding: the selection is restored, or moved when hidden.
        if (IsShowingResults && selected is not null)
        {
            SelectedResult = shown.Contains(selected) ? selected : shown.FirstOrDefault();
        }

        int hiddenByFilters = _allResults.Count(result => !DiscoverFilter.Matches(result.Summary, filters, _capacity));
        int hidden = _allResults.Count - shown.Count - hiddenByFilters;
        if (_allResults.Count > 0)
        {
            Status = shown.Count == 1 ? "1 model" : $"{shown.Count} models";
        }

        var notes = new List<string>();
        if (hidden > 0)
        {
            notes.Add(hidden == 1 ? "1 model is hidden because it is not supported." : $"{hidden} models are hidden because they are not supported.");
        }

        if (hiddenByFilters > 0)
        {
            notes.Add(hiddenByFilters == 1 ? "1 model does not match the filters." : $"{hiddenByFilters} models do not match the filters.");
        }

        HiddenResultsText = string.Join(" ", notes);

        bool IsShown(HubResultViewModel result)
            => DiscoverFilter.Matches(result.Summary, filters, _capacity) && (!ShowOnlySupported || result.Compatibility.IsGood);
    }

    /// <summary>
    /// Shows the quantizations that pass <see cref="ShowOnlySupported"/> (those still being checked included), in their order, and moves
    /// the selection to a shown one when the selected one is hidden.
    /// </summary>
    private void ApplyVariantFilter()
    {
        var shown = _allVariants.Where(variant => !ShowOnlySupported || variant.Compatibility is null || variant.Compatibility.IsGood).ToList();
        if (!shown.SequenceEqual(Variants))
        {
            var selected = SelectedVariant;
            Variants.Clear();
            foreach (var variant in shown)
            {
                Variants.Add(variant);
            }

            SelectedVariant = selected is not null && shown.Contains(selected)
                ? selected
                : shown.FirstOrDefault(variant => variant.Recommended) ?? shown.FirstOrDefault();
        }

        int hidden = _allVariants.Count - shown.Count;
        HiddenVariantsText = hidden == 0 ? "" : hidden == 1
            ? "1 quantization is hidden because it is not supported."
            : $"{hidden} quantizations are hidden because they are not supported.";
    }

    private void OnVariantChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName == nameof(VariantViewModel.Compatibility) && ShowOnlySupported)
        {
            ApplyVariantFilter();
        }
    }

    partial void OnSelectedVariantChanged(VariantViewModel? value)
    {
        if (value?.Compatibility?.Compatibility is { WeightsChecked: false } verdict && !verdict.IsUnsupported
            && verdict.State != CompatibilityState.Projector && _openDetails is { } details && _details is { } cancellation)
        {
            CompatibilityCheck = CheckWeightFormatsAsync(details, value, cancellation.Token);
        }
    }

    [RelayCommand]
    private async Task SearchAsync()
    {
        if (IsHighlightMode)
        {
            return;
        }

        IsSearching = true;
        IsShowingHighlights = false;
        Status = "Searching…";
        try
        {
            var filters = Filters.Current;
            _searched = filters;
            if (filters.FitsThisMachine)
            {
                await UpdateCapacityAsync();
            }

            var results = await _services.Hub.SearchAsync(DiscoverFilter.ToQuery(filters, Query.Trim(), ResultLimit), CancellationToken.None);
            SelectedResult = null;
            _allResults.Clear();
            // Results that are likely not to import go last; the Hub's order (the chosen sort) is kept otherwise.
            _allResults.AddRange(results.Select(result => new HubResultViewModel(result)).OrderBy(result => result.Compatibility.IsBad));
            ApplyResultFilter();

            Status = Results.Count == 0 ? "No models found." : Results.Count == 1 ? "1 model" : $"{Results.Count} models";
            SelectedResult = Results.FirstOrDefault();
        }
        catch (Exception error) when (error is HubException or HttpRequestException or TaskCanceledException)
        {
            Status = error.Message;
        }
        finally
        {
            IsSearching = false;
        }
    }

    private bool CanDownload() => SelectedResult is not null && SelectedVariant is not null;

    [RelayCommand(CanExecute = nameof(CanDownload))]
    private async Task DownloadAsync()
    {
        var result = SelectedResult!;
        var variant = SelectedVariant!;
        if (!await _services.Licenses.RequestAsync(LicenseNotices.TermsFor(result.Id, result.License, result.Summary.Gated)))
        {
            return;
        }

        var tag = variant.Info.Tag;
        var reference = new ModelReference(result.Id, tag.Length == 0 ? null : tag);
        _downloads.Enqueue(reference, variant.Info.Bytes, result.HasKind ? result.Kind.Id : null);
    }

    [RelayCommand]
    private void OpenOnHub()
    {
        if (SelectedResult is not null)
        {
            _services.Platform.Open(new Uri(HubClientOptions.DefaultEndpoint, SelectedResult.Id).ToString());
        }
    }

    private async Task LoadDetailsAsync(HubResultViewModel result)
    {
        _details?.Cancel();
        var cancellation = new CancellationTokenSource();
        _details = cancellation;
        IsLoadingDetails = true;
        foreach (var variant in _allVariants)
        {
            variant.PropertyChanged -= OnVariantChanged;
        }

        _allVariants.Clear();
        Variants.Clear();
        HiddenVariantsText = "";
        Files.Clear();
        Components.Clear();
        IsMultiComponent = false;
        IsPulledAsOne = false;
        NoDownloadReason = "";
        GatedNotice = result.Gated ? GatedText() : "";
        DetailFormat = null;
        _openDetails = null;
        try
        {
            var details = await _services.Hub.GetDetailsAsync(result.Id, cancellation.Token);
            if (cancellation.IsCancellationRequested)
            {
                return;
            }

            var stored = _storedNames();
            foreach (var variant in details.Variants)
            {
                var name = variant.Tag.Length == 0 ? result.Id : $"{result.Id}:{variant.Tag}";
                var item = new VariantViewModel(variant, stored.Contains(name));
                item.PropertyChanged += OnVariantChanged;
                _allVariants.Add(item);
                Variants.Add(item);
            }

            foreach (var file in details.Repository.Files.OrderBy(file => file.Path, StringComparer.Ordinal))
            {
                Files.Add($"{file.Path}  ({Studio.Models.ByteFormat.Format(file.Size)})");
            }

            DescribeLayout(result, details);
            SelectedVariant = Variants.FirstOrDefault(variant => variant.Recommended) ?? Variants.FirstOrDefault();
            FitEstimate = EstimateFitAsync(result, details, [.. _allVariants], cancellation.Token);
        }
        catch (Exception error) when (error is HubException or HttpRequestException or OperationCanceledException)
        {
            if (!cancellation.IsCancellationRequested)
            {
                DetailFormat = error.Message;
            }
        }
        finally
        {
            if (_details == cancellation)
            {
                IsLoadingDetails = false;
            }
        }
    }

    /// <summary>
    /// Fills the details of a repository that cannot be pulled: its components shown as one model (a Diffusers pipeline's folders,
    /// whisper.cpp's model files) and why it cannot be downloaded (its kind or layout is not supported yet).
    /// </summary>
    private void DescribeLayout(HubResultViewModel result, RepositoryDetails details)
    {
        if (details.Format is not null)
        {
            DetailFormat = details.Format.ToString();
            if (details.Layout is not null)
            {
                // A multi-component repository that is pulled as one model: its parts are shown beside the variants (weight precision).
                foreach (var part in details.Parts)
                {
                    Components.Add(new ComponentViewModel(part));
                }

                IsMultiComponent = true;
                IsPulledAsOne = true;
            }

            return;
        }

        var kind = ModelKindCatalog.For(result.Summary.Task);
        if (details.Layout is { } layout)
        {
            foreach (var part in details.Parts)
            {
                Components.Add(new ComponentViewModel(part));
            }

            IsMultiComponent = true;
            long total = details.Parts.Sum(part => part.Bytes);
            DetailFormat = $"{ModelFormatTitles.Title(layout)} · {(Components.Count == 1 ? "1 part" : $"{Components.Count} parts")} · {Studio.Models.ByteFormat.Format(total)}";
            NoDownloadReason = details.UnsupportedReason.Length > 0 ? details.UnsupportedReason
                : kind.Runnable
                ? $"Sharplight cannot download {ModelFormatTitles.Title(layout)} repositories yet."
                : $"Not yet supported: {kind.Title.ToLowerInvariant()}. Sharplight cannot download or run these models yet; the parts are listed as the one model they form.";
            return;
        }

        DetailFormat = "No supported model files";
        NoDownloadReason = kind.Runnable ? "" : $"Not yet supported: {kind.Title.ToLowerInvariant()}.";
    }

    /// <summary>What to do about a gated repository, depending on whether a token is set.</summary>
    private string GatedText()
    {
        bool hasToken = !string.IsNullOrWhiteSpace(_services.Settings.Current.HuggingFaceToken) || !string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable(HubClientOptions.TokenVariable));
        return hasToken
            ? "Gated: accept the terms on Hugging Face once; your access token is used for the download."
            : "Gated: accept the terms on Hugging Face and add your access token in Settings before downloading.";
    }

    /// <summary>
    /// Reads the recommended variant's metadata (a GGUF header and tensor table, the checkpoint's configuration, an ONNX model's opsets)
    /// and badges every variant with its likely compatibility and its approximate fit for the default load settings: the download size
    /// as weights and the KV cache from the dimensions read, when they can be read. The other variants share the architecture's
    /// verdict until they are selected, which reads their own weight formats.
    /// </summary>
    private async Task EstimateFitAsync(HubResultViewModel result, RepositoryDetails details, IReadOnlyList<VariantViewModel> variants, CancellationToken cancellationToken)
    {
        if (variants.Count == 0)
        {
            IsEstimatingFit = false;
            if (details.Format is null && details.Layout is null && result.Compatibility.Compatibility.State != CompatibilityState.UnsupportedKind)
            {
                result.Compatibility = new CompatibilityBadgeViewModel(ModelCompatibility.Unknown("The repository has no GGUF, safetensors or ONNX model files."));
            }

            return;
        }

        IsEstimatingFit = true;
        try
        {
            var source = variants.FirstOrDefault(variant => variant.Recommended) ?? variants[0];
            var probe = await _services.Hub.ProbeAsync(details, source.Info, readWeightFormats: true, cancellationToken);
            if (cancellationToken.IsCancellationRequested)
            {
                return;
            }

            foreach (var variant in variants)
            {
                variant.Compatibility = new CompatibilityBadgeViewModel(ReferenceEquals(variant, source) ? probe.Compatibility : probe.ArchitectureCompatibility);
            }

            if (probe.ArchitectureCompatibility.State != CompatibilityState.Unknown)
            {
                result.Compatibility = new CompatibilityBadgeViewModel(probe.ArchitectureCompatibility);
            }

            _openDetails = details;
            OnPropertyChanged(nameof(SelectedCompatibility));
            if (SelectedVariant is { } selected && !ReferenceEquals(selected, source))
            {
                OnSelectedVariantChanged(selected);
            }

            var shape = probe.Shape;
            var memory = await _advisor.GetMemoryAsync(cancellationToken);
            var settings = _services.Settings.Current.DefaultLoad;
            int context = DownloadEstimate.Context(shape, settings);
            foreach (var variant in variants)
            {
                var footprint = DownloadEstimate.Estimate(variant.Info.Bytes, shape, settings);
                var (report, split) = _advisor.PlanDownload(footprint, memory);
                variant.Fit = new FitBadgeViewModel(report, split, shape is not null, context);
            }

            OnPropertyChanged(nameof(SelectedFit));
            OnPropertyChanged(nameof(HasFitWarning));
            OnPropertyChanged(nameof(HasFitOk));
        }
        catch (Exception error) when (error is not OutOfMemoryException)
        {
            // No estimate (cancelled, or the devices or the Hub could not be asked): the variants go without a badge.
        }
        finally
        {
            if (!cancellationToken.IsCancellationRequested)
            {
                IsEstimatingFit = false;
            }
        }
    }

    /// <summary>Reads a selected variant's own weight formats (its GGUF tensor table) and updates its badge.</summary>
    private async Task CheckWeightFormatsAsync(RepositoryDetails details, VariantViewModel variant, CancellationToken cancellationToken)
    {
        try
        {
            var probe = await _services.Hub.ProbeAsync(details, variant.Info, readWeightFormats: true, cancellationToken);
            if (!cancellationToken.IsCancellationRequested)
            {
                variant.Compatibility = new CompatibilityBadgeViewModel(probe.Compatibility);
                OnPropertyChanged(nameof(SelectedCompatibility));
            }
        }
        catch (OperationCanceledException)
        {
            // Another repository was opened.
        }
    }
}
