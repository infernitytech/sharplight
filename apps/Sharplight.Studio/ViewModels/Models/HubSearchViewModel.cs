using System.Collections.ObjectModel;
using System.ComponentModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Hub;
using Sharplight.Studio.Services;
using Sharplight.Studio.Services.Compatibility;
using Sharplight.Studio.Services.Fit;
using Sharplight.Studio.Services.Hub;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>Searching the Hugging Face Hub and choosing a variant to download.</summary>
internal sealed partial class HubSearchViewModel : ObservableObject
{
    private const int ResultLimit = 40;

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

    public HubSearchViewModel(StudioServices services, DownloadsViewModel downloads, Func<IReadOnlySet<string>> storedNames)
    {
        _services = services;
        _downloads = downloads;
        _storedNames = storedNames;
        _advisor = new FitAdvisor(services.Runtime, services.Settings);
        SelectedFormat = FormatFilter.All[1];
        ShowOnlySupported = services.Settings.Current.ShowOnlySupported;
    }

    /// <summary>Search text.</summary>
    [ObservableProperty]
    public partial string Query { get; set; } = "";

    /// <summary>Format filter choices.</summary>
    public IReadOnlyList<FormatFilter> Formats => FormatFilter.All;

    /// <summary>Chosen format filter.</summary>
    [ObservableProperty]
    public partial FormatFilter SelectedFormat { get; set; }

    /// <summary>Hide repositories that need accepting a license and a token.</summary>
    [ObservableProperty]
    public partial bool HideGated { get; set; }

    /// <summary>
    /// List only repositories and quantizations whose badge is <c>Supported</c> (saved in the settings). Quantizations whose metadata is
    /// still being read stay listed until their badge is known.
    /// </summary>
    [ObservableProperty]
    public partial bool ShowOnlySupported { get; set; }

    /// <summary>The search results shown (all of them, or the supported ones with <see cref="ShowOnlySupported"/>).</summary>
    public ObservableCollection<HubResultViewModel> Results { get; } = [];

    /// <summary>How many search results <see cref="ShowOnlySupported"/> hides, as a sentence; empty when none.</summary>
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

    private const string DefaultStatus = "Search Hugging Face for GGUF, safetensors or ONNX models.";

    /// <summary>Format the selected repository is pulled as.</summary>
    [ObservableProperty]
    public partial string? DetailFormat { get; set; }

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
    }

    partial void OnQueryChanged(string value)
    {
        if (value.Trim().Length == 0 && !IsShowingHighlights)
        {
            IsShowingHighlights = true;
            SelectedResult = null;
            Status = DefaultStatus;
        }
    }

    /// <summary>
    /// Fetches the highlighted models once (the catalog caches them for a few hours, so opening Discover again does not ask the Hub).
    /// </summary>
    public Task LoadHighlightsAsync()
    {
        if (_highlightsLoad is null || _highlightsFailed)
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
        IsLoadingHighlights = true;
        HighlightsStatus = "";
        try
        {
            var sections = await _services.Hub.GetHighlightsAsync(CancellationToken.None);
            Highlights.Clear();
            foreach (var section in sections.Where(section => section.Models.Count > 0))
            {
                Highlights.Add(new HighlightSectionViewModel(section.Kind, [.. section.Models.Select(model => new HubResultViewModel(model))]));
            }

            if (Highlights.Count == 0)
            {
                HighlightsStatus = "No highlighted models right now. Search to find models.";
            }
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

    /// <summary>Shows the results that pass <see cref="ShowOnlySupported"/>, in their order; a hidden selection moves to the first shown.</summary>
    private void ApplyResultFilter()
    {
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

        int hidden = _allResults.Count - shown.Count;
        if (_allResults.Count > 0)
        {
            Status = shown.Count == 1 ? "1 model" : $"{shown.Count} models";
        }

        HiddenResultsText = hidden == 0 ? "" : hidden == 1 ? "1 model is hidden because it is not supported." : $"{hidden} models are hidden because they are not supported.";

        bool IsShown(HubResultViewModel result) => !ShowOnlySupported || result.Compatibility.IsGood;
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

    partial void OnHideGatedChanged(bool value) => _ = SearchAsync();

    partial void OnSelectedVariantChanged(VariantViewModel? value)
    {
        if (value?.Compatibility?.Compatibility is { WeightsChecked: false } verdict && !verdict.IsUnsupported
            && verdict.State != CompatibilityState.Projector && _openDetails is { } details && _details is { } cancellation)
        {
            CompatibilityCheck = CheckWeightFormatsAsync(details, value, cancellation.Token);
        }
    }

    partial void OnSelectedFormatChanged(FormatFilter value) => _ = SearchAsync();

    [RelayCommand]
    private async Task SearchAsync()
    {
        if (Query.Trim().Length == 0)
        {
            return;
        }

        IsSearching = true;
        IsShowingHighlights = false;
        Status = "Searching…";
        try
        {
            var results = await _services.Hub.SearchAsync(new HubSearchQuery { Text = Query.Trim(), Format = SelectedFormat.Format, Limit = ResultLimit }, CancellationToken.None);
            SelectedResult = null;
            _allResults.Clear();
            // Results that are likely not to import go last; the Hub's order (downloads) is kept otherwise.
            _allResults.AddRange(results.Where(result => !HideGated || !result.Gated).Select(result => new HubResultViewModel(result))
                .OrderBy(result => result.Compatibility.IsBad));
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
    private void Download()
    {
        var tag = SelectedVariant!.Info.Tag;
        var reference = new ModelReference(SelectedResult!.Id, tag.Length == 0 ? null : tag);
        _downloads.Enqueue(reference, SelectedVariant.Info.Bytes);
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

            DetailFormat = details.Format?.ToString() ?? "No supported model files";
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
            if (details.Format is null)
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
