using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services;
using Sharplight.Studio.Services.Maintenance;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>
/// The Storage tab of the models page: what Studio uses in every storage location (models, precompiled kernels and device stages, media
/// caches, downloads in progress, logs, outputs), the biggest models, and the cleanup of caches, partial downloads and unused models, each
/// asked for with the space it frees. The measuring and deleting run off the UI thread with progress and cancellation.
/// </summary>
internal sealed partial class StorageViewModel : ObservableObject
{
    private readonly StudioServices _services;
    private readonly DownloadsViewModel _downloads;
    private readonly LibraryViewModel _library;
    private readonly FeedCatalogViewModel _feed;
    private StorageOverview? _overview;

    public StorageViewModel(StudioServices services, DownloadsViewModel downloads, LibraryViewModel library, FeedCatalogViewModel feed)
    {
        _services = services;
        _downloads = downloads;
        _library = library;
        _feed = feed;
        Operation = new FileOperationViewModel(services.Dispatcher);
        library.PropertyChanged += (_, args) =>
        {
            if (args.PropertyName is nameof(LibraryViewModel.UnusedSuggestion))
            {
                OnPropertyChanged(nameof(UnusedSuggestion));
                OnPropertyChanged(nameof(HasUnusedSuggestion));
            }
        };
    }

    /// <summary>Raised when the user asks to review the unused models (the page shows the library with them ticked).</summary>
    public event EventHandler? ReviewUnusedRequested;

    /// <summary>The measuring or cleanup running now.</summary>
    public FileOperationViewModel Operation { get; }

    /// <summary>The storage locations with what is in them.</summary>
    public ObservableCollection<StorageLocationViewModel> Locations { get; } = [];

    /// <summary>The biggest models of the library.</summary>
    public ObservableCollection<LargestModelViewModel> LargestModels { get; } = [];

    /// <summary>The question waiting for an answer, or null.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasConfirmation))]
    public partial ConfirmationViewModel? Confirmation { get; set; }

    /// <summary>A question is shown.</summary>
    public bool HasConfirmation => Confirmation is not null;

    /// <summary>The result of the last action or a note.</summary>
    [ObservableProperty]
    public partial string Status { get; set; } = "";

    /// <summary>The page has measured the locations at least once.</summary>
    [ObservableProperty]
    public partial bool IsMeasured { get; private set; }

    /// <summary>Bytes of the kernel cache (compiled kernels, CUDA PTX and cubin, SPIR-V device stages).</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(KernelCacheText))]
    public partial long KernelCacheBytes { get; private set; }

    /// <summary>Bytes of the shader source cache.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(ShaderCacheText))]
    public partial long ShaderCacheBytes { get; private set; }

    /// <summary>Bytes of the media and other caches.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(MediaCacheText))]
    public partial long MediaCacheBytes { get; private set; }

    /// <summary>Bytes of the partial downloads.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(PartialText), nameof(HasPartialDownloads))]
    public partial long PartialBytes { get; private set; }

    /// <summary>How many partial downloads there are.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(PartialText), nameof(HasPartialDownloads))]
    public partial int PartialCount { get; private set; }

    /// <summary><c>Precompiled kernels · 1.2 GB</c>.</summary>
    public string KernelCacheText => ByteFormat.Format(KernelCacheBytes);

    /// <summary>The shader cache size.</summary>
    public string ShaderCacheText => ByteFormat.Format(ShaderCacheBytes);

    /// <summary>The media cache size.</summary>
    public string MediaCacheText => ByteFormat.Format(MediaCacheBytes);

    /// <summary>There are failed or partial downloads to remove.</summary>
    public bool HasPartialDownloads => PartialCount > 0;

    /// <summary><c>2 failed or partial downloads · 3.1 GB</c>.</summary>
    public string PartialText => PartialCount == 0 ? "No failed or partial downloads." : $"{ByteFormat.Count(PartialCount, "failed or partial download", "failed or partial downloads")} · {ByteFormat.Format(PartialBytes)}";

    /// <summary>Models not used for 90 days (from the library).</summary>
    public bool HasUnusedSuggestion => _library.HasUnusedSuggestion;

    /// <summary>The suggestion text.</summary>
    public string UnusedSuggestion => _library.UnusedSuggestion;

    /// <summary>Measures every location again (off the UI thread).</summary>
    [RelayCommand]
    public async Task RefreshAsync()
    {
        if (Operation.IsRunning)
        {
            return;
        }

        StorageOverview? overview = null;
        bool completed = await Operation.RunAsync("Measuring", async (progress, cancellationToken) =>
            overview = await _services.Maintenance.GetOverviewAsync(progress, cancellationToken).ConfigureAwait(false));
        if (!completed || overview is null)
        {
            Status = "Measuring was cancelled.";
            return;
        }

        Apply(overview);
    }

    private void Apply(StorageOverview overview)
    {
        _overview = overview;
        Locations.Clear();
        foreach (var location in overview.Locations)
        {
            Locations.Add(new StorageLocationViewModel(location));
        }

        long Used(string key) => overview.Locations.SelectMany(location => location.Items).Where(item => item.Key == key).Sum(item => item.Bytes);
        KernelCacheBytes = Used(UsageKeys.KernelCache);
        ShaderCacheBytes = Used(UsageKeys.ShaderCache);
        MediaCacheBytes = Used(UsageKeys.MediaCache);
        PartialBytes = overview.PartialBytes;
        PartialCount = overview.PartialDownloads.Count;
        RefreshLargest();
        IsMeasured = true;
    }

    /// <summary>Lists the biggest models of the library again.</summary>
    public void RefreshLargest()
    {
        LargestModels.Clear();
        var top = _library.Entries.OrderByDescending(entry => entry.SizeBytes).Take(8).ToList();
        foreach (var entry in top)
        {
            LargestModels.Add(new LargestModelViewModel(entry, top[0].SizeBytes));
        }
    }

    [RelayCommand]
    private void ReviewUnused()
    {
        _library.ReviewUnused();
        ReviewUnusedRequested?.Invoke(this, EventArgs.Empty);
    }

    [RelayCommand]
    private Task ClearKernelCacheAsync() => RequestClearAsync(CacheKinds.Kernels, "the precompiled kernel cache (compiled kernels, CUDA PTX and cubin, SPIR-V device stages)",
        "They are compiled again the next time a model needs them; models precompiled into their package are not affected.");

    [RelayCommand]
    private Task ClearShaderCacheAsync() => RequestClearAsync(CacheKinds.Shaders, "the extracted shader sources", "They are extracted again when needed.");

    [RelayCommand]
    private Task ClearMediaCacheAsync() => RequestClearAsync(CacheKinds.Media, "the media and other caches", "They are rebuilt when needed.");

    [RelayCommand]
    private Task ClearAllCachesAsync() => RequestClearAsync(CacheKinds.All, "all caches", "They are rebuilt when needed.");

    /// <summary>Asks before clearing <paramref name="kinds"/>, naming the space it frees (measured again first).</summary>
    public async Task RequestClearAsync(CacheKinds kinds, string what, string consequence)
    {
        if (Operation.IsRunning)
        {
            return;
        }

        long freed = await _services.Maintenance.MeasureCachesAsync(kinds, CancellationToken.None);
        if (freed == 0)
        {
            Status = $"Nothing to clear in {what}.";
            return;
        }

        Ask($"Clear {what}? This frees {ByteFormat.Format(freed)}. {consequence}", "Clear", () => ClearAsync(kinds));
    }

    private async Task ClearAsync(CacheKinds kinds)
    {
        long freed = 0;
        bool completed = await Operation.RunAsync("Clearing caches", async (progress, cancellationToken) =>
            freed = await _services.Maintenance.ClearCachesAsync(kinds, progress, cancellationToken).ConfigureAwait(false));
        Status = $"Cleared caches, freed {ByteFormat.Format(freed)}." + (completed ? "" : " Cancelled.");
        await RefreshAsync();
    }

    /// <summary>
    /// Asks before removing the failed and partial downloads. Refused while a download is running: its partial files are in use and a
    /// cancelled download keeps them to resume.
    /// </summary>
    [RelayCommand]
    public async Task RemovePartialDownloadsAsync()
    {
        if (Operation.IsRunning)
        {
            return;
        }

        if (_overview is null || _overview.PartialDownloads.Count == 0)
        {
            Status = "No failed or partial downloads.";
            return;
        }

        if (_downloads.ActiveCount > 0 || _feed.ActiveInstalls > 0)
        {
            Status = "A download is running. Pause or cancel it first: its partial files are in use.";
            return;
        }

        var items = _overview.PartialDownloads;
        Ask($"Remove {ByteFormat.Count(items.Count, "failed or partial download", "failed or partial downloads")}? This frees {ByteFormat.Format(items.Sum(item => item.Bytes))}. A download that was paused or cancelled cannot resume afterwards and starts again.",
            "Remove", () => RemoveAsync(items));
    }

    private async Task RemoveAsync(IReadOnlyList<PartialDownload> items)
    {
        long freed = 0;
        bool completed = await Operation.RunAsync("Removing partial downloads", async (_, cancellationToken) =>
            freed = await _services.Maintenance.RemovePartialDownloadsAsync(items, cancellationToken).ConfigureAwait(false));
        Status = $"Removed partial downloads, freed {ByteFormat.Format(freed)}." + (completed ? "" : " Cancelled.");
        await RefreshAsync();
    }

    private void Ask(string message, string confirmText, Func<Task> action)
    {
        Confirmation = new ConfirmationViewModel(message, confirmText, isDestructive: true, action, answered =>
        {
            if (ReferenceEquals(Confirmation, answered))
            {
                Confirmation = null;
            }
        });
    }
}
