using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Hub;
using Sharplight.Studio.Services;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>Searching the Hugging Face Hub and choosing a variant to download.</summary>
internal sealed partial class HubSearchViewModel : ObservableObject
{
    private const int ResultLimit = 40;

    private readonly StudioServices _services;
    private readonly DownloadsViewModel _downloads;
    private readonly Func<IReadOnlySet<string>> _storedNames;
    private CancellationTokenSource? _details;

    public HubSearchViewModel(StudioServices services, DownloadsViewModel downloads, Func<IReadOnlySet<string>> storedNames)
    {
        _services = services;
        _downloads = downloads;
        _storedNames = storedNames;
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
    public partial VariantViewModel? SelectedVariant { get; set; }

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
            foreach (var result in results.Where(result => !HideGated || !result.Gated))
            {
                Results.Add(new HubResultViewModel(result));
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
}
