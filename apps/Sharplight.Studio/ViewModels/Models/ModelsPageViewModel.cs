using CommunityToolkit.Mvvm.ComponentModel;
using Sharplight.Studio.Services;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>The models page: Discover (Hub search), Library, Resident models and Downloads.</summary>
internal sealed partial class ModelsPageViewModel : PageViewModel
{
    /// <summary>Tab index of Discover.</summary>
    public const int DiscoverTab = 0;

    /// <summary>Tab index of the library.</summary>
    public const int LibraryTab = 1;

    /// <summary>Tab index of the resident models.</summary>
    public const int ResidentTab = 2;

    /// <summary>Tab index of the downloads.</summary>
    public const int DownloadsTab = 3;

    /// <summary>Tab index of the storage overview and cleanup.</summary>
    public const int StorageTab = 4;

    private readonly StudioServices _services;
    private readonly bool _backgroundChecks;
    private readonly ModelUpdateWatcher _watcher;
    private bool _loaded;
    private bool _active;
    private IDisposable? _memoryWatch;

    /// <param name="services">The services.</param>
    /// <param name="navigator">Opens other pages.</param>
    /// <param name="time">The clock (tests set it).</param>
    /// <param name="backgroundChecks">Start the periodic check for newer model versions when the settings turn it on (tests turn it off and call the watcher).</param>
    public ModelsPageViewModel(StudioServices services, INavigator navigator, TimeProvider? time = null, bool backgroundChecks = true)
    {
        _services = services;
        _backgroundChecks = backgroundChecks;
        Downloads = new DownloadsViewModel(services);
        Library = new LibraryViewModel(services, navigator, time);
        Resident = new ResidentModelsViewModel(services, navigator);
        Discover = new HubSearchViewModel(services, Downloads, () => Library.StoredNames, () => Library.StoredManifests);
        Library.Feed = Discover.Feed;
        Storage = new StorageViewModel(services, Downloads, Library, Discover.Feed);
        Storage.ReviewUnusedRequested += (_, _) => SelectedTab = LibraryTab;
        Library.UpdatesChanged += (_, _) => UpdateBadge();
        _watcher = new ModelUpdateWatcher(services, CheckUpdatesAsync, time);
        services.Settings.Changed += (_, _) => services.Dispatcher.Post(StartOrStopWatcher);
        Downloads.ModelStored += async (_, _) =>
        {
            await Library.RefreshAsync();
            Discover.Verified.RefreshInstalled();
        };
        Discover.Feed.Installed += async (_, _) =>
        {
            await Library.RefreshAsync();
            Discover.Feed.RefreshInstalled();
            Library.ApplyFeedUpdates();
        };
        Discover.Feed.ListingChanged += (_, _) => Library.ApplyFeedUpdates();
        Downloads.Enqueued += (_, _) => SelectedTab = DownloadsTab;
        Downloads.PropertyChanged += (_, args) =>
        {
            if (args.PropertyName == nameof(DownloadsViewModel.ActiveCount))
            {
                UpdateBadge();
            }
        };
        Downloads.Restore();
        StartOrStopWatcher();
    }

    /// <inheritdoc/>
    public override string Title => "Models";

    /// <inheritdoc/>
    public override string Icon => "IconModels";

    /// <summary>Hub search.</summary>
    public HubSearchViewModel Discover { get; }

    /// <summary>Local models.</summary>
    public LibraryViewModel Library { get; }

    /// <summary>Downloads.</summary>
    public DownloadsViewModel Downloads { get; }

    /// <summary>Resident models.</summary>
    public ResidentModelsViewModel Resident { get; }

    /// <summary>Disk usage and cleanup.</summary>
    public StorageViewModel Storage { get; }

    /// <summary>The check for newer model versions (Settings &gt; Models).</summary>
    public ModelUpdateWatcher UpdateWatcher => _watcher;

    /// <summary>The visible tab.</summary>
    [ObservableProperty]
    public partial int SelectedTab { get; set; } = LibraryTab;

    partial void OnSelectedTabChanged(int value)
    {
        WatchMemory();
        if (value == DiscoverTab)
        {
            _ = Discover.LoadHighlightsAsync();
        }
        else if (value == StorageTab)
        {
            _ = Storage.RefreshAsync();
        }
    }

    /// <summary>Downloads still running plus the models that have a newer version.</summary>
    private void UpdateBadge() => Badge = Downloads.ActiveCount + Library.UpdateCount;

    private void StartOrStopWatcher()
    {
        if (_backgroundChecks && _services.Settings.Current.CheckModelUpdates)
        {
            _watcher.Start();
        }
        else
        {
            _watcher.Stop();
        }
    }

    private async Task<int> CheckUpdatesAsync()
    {
        await EnsureLoadedAsync();
        await Discover.Feed.RefreshAsync();
        return Library.UpdateCount;
    }

    private async Task EnsureLoadedAsync()
    {
        if (!_loaded)
        {
            _loaded = true;
            await Library.RefreshAsync();
        }
    }

    /// <summary>
    /// While Discover is visible, the devices' memory is read again every few seconds and after loads and unloads, so its fit badges
    /// follow (the load panel watches by itself).
    /// </summary>
    private void WatchMemory()
    {
        if (_active && SelectedTab == DiscoverTab)
        {
            _memoryWatch ??= _services.Memory.Watch();
        }
        else
        {
            _memoryWatch?.Dispose();
            _memoryWatch = null;
        }
    }

    /// <inheritdoc/>
    public override void Deactivate()
    {
        _active = false;
        WatchMemory();
    }

    /// <inheritdoc/>
    public override async Task ActivateAsync()
    {
        _active = true;
        WatchMemory();
        if (!_loaded)
        {
            await EnsureLoadedAsync();
            // The listing tells the library which installed packages have a newer version.
            _ = Discover.Feed.LoadAsync();
            if (Library.IsEmpty)
            {
                SelectedTab = DiscoverTab;
            }
        }
    }
}
