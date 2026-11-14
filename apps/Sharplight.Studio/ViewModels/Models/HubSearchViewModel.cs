using System.Collections.ObjectModel;
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
    private CancellationTokenSource? _details;
    private RepositoryDetails? _openDetails;

    public HubSearchViewModel(StudioServices services, DownloadsViewModel downloads, Func<IReadOnlySet<string>> storedNames)
    {
        _services = services;
        _downloads = downloads;
        _storedNames = storedNames;
        _advisor = new FitAdvisor(services.Runtime, services.Settings);
        SelectedFormat = FormatFilter.All[1];
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

    /// <summary>Results.</summary>
    public ObservableCollection<HubResultViewModel> Results { get; } = [];

    /// <summary>The result whose details are shown.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasSelection), nameof(SelectedId))]
    public partial HubResultViewModel? SelectedResult { get; set; }

    /// <summary>Variants of the selected repository.</summary>
    public ObservableCollection<VariantViewModel> Variants { get; } = [];

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
    public partial string Status { get; set; } = "Search Hugging Face for GGUF, safetensors or ONNX models.";

    /// <summary>Format the selected repository is pulled as.</summary>
    [ObservableProperty]
    public partial string? DetailFormat { get; set; }

    /// <summary>A result is selected.</summary>
    public bool HasSelection => SelectedResult is not null;

    /// <summary>The selected repository id (empty without a selection).</summary>
    public string SelectedId => SelectedResult?.Id ?? "";

    partial void OnSelectedResultChanged(HubResultViewModel? value)
    {
        if (value is not null)
        {
            _ = LoadDetailsAsync(value);
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
        Status = "Searching…";
        try
        {
            var results = await _services.Hub.SearchAsync(new HubSearchQuery { Text = Query.Trim(), Format = SelectedFormat.Format, Limit = ResultLimit }, CancellationToken.None);
            Results.Clear();
            // Results that are likely not to import go last; the Hub's order (downloads) is kept otherwise.
            foreach (var result in results.Where(result => !HideGated || !result.Gated).Select(result => new HubResultViewModel(result))
                .OrderBy(result => result.Compatibility.IsBad))
            {
                Results.Add(result);
            }

            Status = Results.Count == 0 ? "No models found." : $"{Results.Count} models";
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
        Variants.Clear();
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
                Variants.Add(new VariantViewModel(variant, stored.Contains(name)));
            }

            foreach (var file in details.Repository.Files.OrderBy(file => file.Path, StringComparer.Ordinal))
            {
                Files.Add($"{file.Path}  ({Studio.Models.ByteFormat.Format(file.Size)})");
            }

            DetailFormat = details.Format?.ToString() ?? "No supported model files";
            SelectedVariant = Variants.FirstOrDefault(variant => variant.Recommended) ?? Variants.FirstOrDefault();
            FitEstimate = EstimateFitAsync(result, details, [.. Variants], cancellation.Token);
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
