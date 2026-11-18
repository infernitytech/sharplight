using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services;
using Sharplight.Studio.Services.Compatibility;
using Sharplight.Studio.Services.FirstRun;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.ViewModels.Models;

namespace Sharplight.Studio.ViewModels.Welcome;

/// <summary>
/// The welcome flow: describes the machine, recommends one to three starter models from the model feed, installs the selection through the
/// feed installer, loads the chat model and opens a new chat with an example question. Offline it works from the saved listing or says why
/// it cannot. Opened on the first start (empty library, flow not done yet) and from the chat page's "Get a model".
/// </summary>
internal sealed partial class WelcomePageViewModel : PageViewModel
{
    private readonly StudioServices _services;
    private readonly INavigator _navigator;
    private readonly ModelsPageViewModel _models;
    private readonly MachineProfiler _profiler;
    private readonly StarterRecommender _recommender;
    private Task? _load;
    private string? _loadedId;

    public WelcomePageViewModel(StudioServices services, INavigator navigator, ModelsPageViewModel models, Func<string, long?>? freeDisk = null)
    {
        _services = services;
        _navigator = navigator;
        _models = models;
        _profiler = new MachineProfiler(services.Runtime, freeDisk);
        _recommender = new StarterRecommender(StarterPreferencesLoader.Load(services.StorageLocation.Paths.StarterModelsFile));
        Machine = _profiler.Describe(services.Settings.StoreRoot);
        Items.CollectionChanged += (_, _) => NotifyTotals();
    }

    /// <inheritdoc/>
    public override string Title => "Welcome";

    /// <inheritdoc/>
    public override string Icon => "IconSparkle";

    /// <summary>The machine as detected when the page was created.</summary>
    [ObservableProperty]
    public partial MachineProfile Machine { get; set; }

    /// <summary>The GPU line, e.g. <c>NVIDIA RTX 4090 · 24 GB</c> or <c>No GPU found: models run on the CPU</c>.</summary>
    public string GpuText => Machine.HasGpu
        ? $"{Machine.GpuName} · {ByteFormat.Format(Machine.GpuMemoryBytes)}{(Machine.GpuCount > 1 ? $" (1 of {Machine.GpuCount} GPUs)" : "")}"
        : "No GPU found: models run on the CPU";

    /// <summary>The system memory line.</summary>
    public string MemoryText => Machine.SystemMemoryBytes > 0 ? ByteFormat.Format(Machine.SystemMemoryBytes) : "Unknown";

    /// <summary>The free disk line for the model store.</summary>
    public string DiskText => Machine.FreeDiskBytes is { } free ? $"{ByteFormat.Format(free)} free where models are stored" : "Unknown";

    /// <summary>The recommended packages.</summary>
    public ObservableCollection<StarterItemViewModel> Items { get; } = [];

    /// <summary>There is something to install.</summary>
    public bool HasItems => Items.Count > 0;

    /// <summary>The feed is being read.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(ShowEmpty))]
    public partial bool IsLoading { get; set; }

    /// <summary>The recommendation, or null before the listing was read.</summary>
    [ObservableProperty]
    public partial StarterRecommendation? Recommendation { get; set; }

    /// <summary>Why nothing is recommended, or a note about the list (saved copy), else empty.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasMessage), nameof(ShowEmpty))]
    public partial string Message { get; set; } = "";

    /// <summary>A note is shown.</summary>
    public bool HasMessage => Message.Length > 0;

    /// <summary>The list could not be read or nothing fits: the page shows the message instead of models.</summary>
    public bool ShowEmpty => !IsLoading && !HasItems && HasMessage;

    /// <summary>The recommendation is sized to the CPU.</summary>
    public bool IsCpuOnly => Recommendation?.CpuOnly == true;

    /// <summary>The list is the saved copy because the feed could not be reached.</summary>
    [ObservableProperty]
    public partial bool IsOffline { get; set; }

    /// <summary>The install and load are running.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanStart), nameof(ShowProgress))]
    [NotifyCanExecuteChangedFor(nameof(GetStartedCommand))]
    public partial bool IsWorking { get; set; }

    /// <summary>The install ended with an error (the button reads <c>Try again</c>).</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasError), nameof(StartText), nameof(ShowProgress))]
    public partial string? Error { get; set; }

    /// <summary>An error is shown.</summary>
    public bool HasError => Error is not null;

    /// <summary>Overall progress of the downloads and the load, 0 to 1.</summary>
    [ObservableProperty]
    public partial double Progress { get; set; }

    /// <summary>What the install is doing.</summary>
    [ObservableProperty]
    public partial string Status { get; set; } = "";

    /// <summary>Progress is shown.</summary>
    public bool ShowProgress => IsWorking || HasError;

    /// <summary>The selection can be installed.</summary>
    public bool CanStart => !IsWorking && Items.Any(item => item.IsSelected);

    /// <summary>The label of the main button.</summary>
    public string StartText => HasError ? "Try again" : "Get started";

    /// <summary>What the selection costs, e.g. <c>Download 1.4 GB · about 2 min</c>.</summary>
    public string TotalText
    {
        get
        {
            long bytes = Items.Where(item => item.IsSelected).Sum(item => item.DownloadBytes);
            return bytes == 0 ? "Nothing to download" : $"Download {ByteFormat.Format(bytes)} · {DurationText(_recommender.EstimateTime(bytes))}";
        }
    }

    /// <summary>The latest read of the feed (for tests).</summary>
    internal Task Loading => _load ?? Task.CompletedTask;

    /// <summary>Starts reading the feed (once, and again when the last read found nothing to offer); the page shows its progress.</summary>
    public override Task ActivateAsync()
    {
        if (_load is null || _load.IsCompleted && !IsWorking && !HasItems)
        {
            _load = LoadAsync();
        }

        return Task.CompletedTask;
    }

    /// <summary>Reads the feed again (after a failure or when the network is back).</summary>
    [RelayCommand]
    public Task RetryAsync() => _load = LoadAsync();

    /// <summary>Installs the selected packages, loads the chat model and opens a new chat.</summary>
    [RelayCommand(CanExecute = nameof(CanStart))]
    public async Task GetStartedAsync()
    {
        var selected = Items.Where(item => item.IsSelected).ToList();
        if (selected.Count == 0 || IsWorking)
        {
            return;
        }

        IsWorking = true;
        Error = null;
        try
        {
            if (!await InstallAsync(selected) || !await LoadChatAsync(selected[0]))
            {
                return;
            }

            _services.Settings.Update(settings => settings with { FirstRunDone = true });
            _navigator.OpenChat(_services.Runtime.LoadedModels.FirstOrDefault(model => model.Id == _loadedId), Recommendation?.ExamplePrompt);
        }
        finally
        {
            IsWorking = false;
            OnPropertyChanged(nameof(CanStart));
        }
    }

    /// <summary>Leaves the flow for Discover; it is not offered again on start.</summary>
    [RelayCommand]
    public void Skip()
    {
        _services.Settings.Update(settings => settings with { FirstRunDone = true });
        _navigator.OpenDiscover();
    }

    private async Task<bool> InstallAsync(List<StarterItemViewModel> selected)
    {
        long total = Math.Max(1, selected.Sum(item => item.DownloadBytes));
        long finished = 0;
        foreach (var item in selected)
        {
            if (item.DownloadBytes == 0)
            {
                item.State = "Ready";
                continue;
            }

            item.State = "Downloading…";
            var install = _models.Discover.Feed.Install(item.Package);
            long baseline = finished;
            install.PropertyChanged += (_, args) =>
            {
                if (args.PropertyName is nameof(FeedInstallViewModel.Progress) or nameof(FeedInstallViewModel.Status))
                {
                    Progress = Math.Min(0.9, (baseline + install.Progress * item.DownloadBytes) / total * 0.9);
                    Status = install.Status;
                }
            };
            await install.Completion;
            if (!install.IsDone)
            {
                item.State = "Failed";
                Error = install.Error ?? install.Status;
                Status = $"Could not install {item.Package.Name}";
                return false;
            }

            item.State = "Ready";
            finished += item.DownloadBytes;
        }

        return true;
    }

    private async Task<bool> LoadChatAsync(StarterItemViewModel chat)
    {
        Status = $"Loading {chat.Package.Name}…";
        Progress = 0.9;
        await _models.Library.RefreshAsync();
        var entries = await _services.Library.ListAsync(CancellationToken.None);
        var entry = entries.FirstOrDefault(candidate => candidate.Manifest?.FeedPackage == chat.Package.Id);
        if (entry is null)
        {
            Error = $"{chat.Package.Name} was installed but is not in the library. Open the Models page to load it.";
            return false;
        }

        var taken = _services.Runtime.LoadedModels.Select(model => model.Id).ToHashSet();
        string id = entry.Id;
        for (int suffix = 2; taken.Contains(id); suffix++)
        {
            id = $"{entry.Id}-{suffix}";
        }

        var load = _services.Settings.Current.DefaultLoad with { DeviceId = Machine.GpuDeviceId ?? _services.Settings.Current.DefaultLoad.DeviceId, Slots = 1 };
        var request = new ModelLoadRequest(id, entry.Path, load, _services.Settings.Current.DisabledFeatures, Capabilities: entry.Capabilities, Kind: entry.Kind);
        try
        {
            using var progress = new DispatchingProgress<LoadStatus>(_services.Dispatcher, update =>
            {
                Progress = 0.9 + 0.1 * Math.Clamp(update.Fraction, 0, 1);
                Status = $"Loading {chat.Package.Name}… {update.Text}".TrimEnd();
            }, static (_, _) => true);
            var model = await Task.Run(() => _services.Runtime.LoadAsync(request, progress, CancellationToken.None), CancellationToken.None);
            progress.Flush();
            _loadedId = model.Id;
            Progress = 1;
            Status = "Ready";
            return true;
        }
        catch (Exception error) when (error is not OutOfMemoryException)
        {
            Error = ImportFailure.Describe(error, error.GetBaseException().Message, entry.Path);
            return false;
        }
    }

    private async Task LoadAsync()
    {
        IsLoading = true;
        Message = "";
        Items.Clear();
        try
        {
            Machine = _profiler.Describe(_services.Settings.StoreRoot);
            OnPropertyChanged(nameof(GpuText));
            OnPropertyChanged(nameof(MemoryText));
            OnPropertyChanged(nameof(DiskText));
            var feed = _models.Discover.Feed;
            await _models.Library.RefreshAsync();
            await feed.RefreshAsync();
            feed.RefreshInstalled();
            IsOffline = feed.IsOffline;
            if (feed.Failure is { } failure)
            {
                Recommendation = null;
                Message = FailureText(failure);
                return;
            }

            var packages = feed.AllPackages;
            var recommendation = _recommender.Recommend([.. packages.Select(package => package.Summary)], Machine, _services.Feed.Environment);
            Recommendation = recommendation;
            foreach (var item in recommendation.Items)
            {
                var package = packages.First(candidate => candidate.Id == item.Package.Id);
                var row = new StarterItemViewModel(item, package, DurationText(item.DownloadTime));
                row.PropertyChanged += (_, args) =>
                {
                    if (args.PropertyName == nameof(StarterItemViewModel.IsSelected))
                    {
                        NotifyTotals();
                    }
                };
                Items.Add(row);
            }

            Message = recommendation.Outcome == StarterOutcome.Recommended
                ? feed.IsOffline ? "The feed could not be reached. These models come from the last saved list; downloading needs the network." : ""
                : recommendation.Message;
        }
        finally
        {
            IsLoading = false;
            OnPropertyChanged(nameof(IsCpuOnly));
            NotifyTotals();
        }
    }

    private void NotifyTotals()
    {
        OnPropertyChanged(nameof(HasItems));
        OnPropertyChanged(nameof(ShowEmpty));
        OnPropertyChanged(nameof(TotalText));
        OnPropertyChanged(nameof(CanStart));
        GetStartedCommand.NotifyCanExecuteChanged();
    }

    private static string FailureText(Services.Feed.FeedFailure failure) => failure.Kind switch
    {
        Services.Feed.FeedFailureKind.Verification => failure.Message,
        _ => $"{failure.Message} Connect to the internet and try again, or skip this step and add a model from a file.",
    };

    internal static string DurationText(TimeSpan time) => time.TotalSeconds < 45 ? "under a minute"
        : time.TotalMinutes < 90 ? $"about {Math.Max(1, (int)Math.Round(time.TotalMinutes))} min"
        : $"about {time.TotalHours:F1} h";
}
