using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Hub;
using Sharplight.ModelFeed.Versioning;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services;
using Sharplight.Studio.Services.Feed;
using Sharplight.Studio.Services.Fit;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>
/// Discover's "Sharplight feed" list: the signed, preconverted <c>.slm</c> packages of the model feed's channel, with their size, version, fit on
/// this machine and verified badge, installed (with their dependencies) into the model store in one click, and the newer versions of the
/// installed ones for the library. The network work runs off the UI thread; offline the last verified listing is used.
/// </summary>
internal sealed partial class FeedCatalogViewModel : ObservableObject
{
    private static readonly FilterOption<string?> s_allKinds = new("All tasks", null);

    private readonly StudioServices _services;
    private readonly Func<IReadOnlyList<ModelManifest>> _storedModels;
    private readonly FitAdvisor _advisor;
    private readonly List<FeedPackageViewModel> _all = [];
    private Task? _load;
    private string _loadedKey = "";
    private int _generation;

    public FeedCatalogViewModel(StudioServices services, Func<IReadOnlyList<ModelManifest>> storedModels)
    {
        _services = services;
        _storedModels = storedModels;
        _advisor = new FitAdvisor(services.Runtime, services.Settings);
        Kinds.Add(s_allKinds);
        SelectedKind = s_allKinds;
        services.Settings.Changed += (_, _) => services.Dispatcher.Post(OnSettingsChanged);
    }

    /// <summary>Raised on the UI thread when a package was stored (the library must list it).</summary>
    public event EventHandler? Installed;

    /// <summary>Raised on the UI thread when the listing was loaded (newer versions of installed packages may be known).</summary>
    public event EventHandler? ListingChanged;

    /// <summary>The tasks the feed has packages for, with their counts, and "All tasks".</summary>
    public ObservableCollection<FilterOption<string?>> Kinds { get; } = [];

    [ObservableProperty]
    public partial FilterOption<string?> SelectedKind { get; set; }

    [ObservableProperty]
    public partial string Query { get; set; } = "";

    /// <summary>Only packages that fit the GPU or system memory.</summary>
    [ObservableProperty]
    public partial bool OnlyFitting { get; set; }

    /// <summary>The packages that match the task, the text and the fit filter.</summary>
    public ObservableCollection<FeedPackageViewModel> Entries { get; } = [];

    /// <summary>Every package of the listing.</summary>
    public IReadOnlyList<FeedPackageViewModel> AllPackages => _all;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasSelection))]
    [NotifyCanExecuteChangedFor(nameof(InstallSelectedCommand))]
    public partial FeedPackageViewModel? Selected { get; set; }

    public bool HasSelection => Selected is not null;

    [ObservableProperty]
    public partial bool IsLoading { get; set; }

    /// <summary>Progress or "nothing matches" text.</summary>
    [ObservableProperty]
    public partial string Status { get; set; } = "";

    /// <summary>The channel, size and age of the listing, e.g. <c>Sharplight feed · stable · 5 packages</c>.</summary>
    [ObservableProperty]
    public partial string ListingText { get; set; } = "";

    /// <summary>The listing is the last verified copy because the feed could not be reached.</summary>
    [ObservableProperty]
    public partial bool IsOffline { get; set; }

    /// <summary>Notes about the offline copy, or empty.</summary>
    [ObservableProperty]
    public partial string OfflineText { get; set; } = "";

    /// <summary>Why nothing is listed (a failed verification, an unreachable feed without cache), or null.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasFailure), nameof(IsVerificationFailure), nameof(FailureText))]
    public partial FeedFailure? Failure { get; set; }

    public bool HasFailure => Failure is not null;

    /// <summary>The failure for display, or empty.</summary>
    public string FailureText => Failure?.Message ?? "";

    /// <summary>The feed failed verification: the page says so prominently and installs nothing.</summary>
    public bool IsVerificationFailure => Failure?.Kind == FeedFailureKind.Verification;

    /// <summary>The installs of this session, newest first.</summary>
    public ObservableCollection<FeedInstallViewModel> Installs { get; } = [];

    /// <summary>How many installs are running.</summary>
    public int ActiveInstalls => Installs.Count(install => install.IsActive);

    /// <summary>How many installs finished successfully and are not dismissed.</summary>
    public int FinishedInstalls => Installs.Count(install => install.IsDone);

    /// <summary>There are finished installs: the page shows one compact summary instead of a card each.</summary>
    public bool HasFinishedInstalls => FinishedInstalls > 0;

    /// <summary>The summary of the finished installs, e.g. <c>Installed 2 packages: Whisper tiny, Qwen3 0.6B</c>.</summary>
    public string FinishedSummary
    {
        get
        {
            var names = Installs.Where(install => install.IsDone).Select(install => install.Name).ToList();
            return names.Count switch
            {
                0 => "",
                1 => $"Installed {names[0]}",
                <= 3 => $"Installed {names.Count} packages: {string.Join(", ", names)}",
                _ => $"Installed {names.Count} packages: {string.Join(", ", names.Take(3))} and {names.Count - 3} more",
            };
        }
    }

    /// <summary>Whether the verified-signature check applied to the listing.</summary>
    [ObservableProperty]
    public partial bool IsListingVerified { get; set; }

    /// <summary>Loads the listing once (off the UI thread).</summary>
    public Task LoadAsync() => _load ??= LoadCoreAsync();

    /// <summary>Reads the feed again.</summary>
    [RelayCommand]
    public Task RefreshAsync() => _load = LoadCoreAsync();

    /// <summary>Marks the packages that are in the model store, and their versions.</summary>
    public void RefreshInstalled()
    {
        var stored = _storedModels();
        foreach (var package in _all)
        {
            package.InstalledVersion = stored.FirstOrDefault(model => model.FeedPackage == package.Id)?.FeedVersion;
        }
    }

    /// <summary>The newer version of the feed package a stored model came from, or null when it is not from the feed or is current.</summary>
    /// <param name="manifest">The store record.</param>
    public string? NewerVersionOf(ModelManifest manifest)
    {
        if (manifest.FeedPackage is null || manifest.FeedVersion is null || !SemanticVersion.TryParse(manifest.FeedVersion, out var have))
        {
            return null;
        }

        var package = _all.FirstOrDefault(candidate => candidate.Id == manifest.FeedPackage);
        return package is { IsInstallable: true } && SemanticVersion.TryParse(package.Version, out var latest) && latest > have ? package.Version : null;
    }

    /// <summary>Starts installing (or updating) a package of the listing.</summary>
    /// <param name="package">The package.</param>
    public FeedInstallViewModel Install(FeedPackageViewModel package)
    {
        var running = Installs.FirstOrDefault(install => install.Id == package.Id && !install.IsDone && (install.IsActive || install.CanRetry));
        if (running is not null)
        {
            _ = running.StartAsync();
            return running;
        }

        var install = new FeedInstallViewModel(package.Id, package.Name, package.Version, package.HasUpdate, _services.Feed, _services.Dispatcher, OnInstalled);
        install.PropertyChanged += (_, args) =>
        {
            if (args.PropertyName == nameof(FeedInstallViewModel.IsActive))
            {
                OnPropertyChanged(nameof(ActiveInstalls));
            }
            else if (args.PropertyName == nameof(FeedInstallViewModel.IsDone))
            {
                NotifyFinishedChanged();
            }
        };
        Installs.Insert(0, install);
        _ = install.StartAsync();
        OnPropertyChanged(nameof(ActiveInstalls));
        return install;
    }

    /// <summary>Starts installing the latest version of a package by id (the library's update action).</summary>
    /// <param name="id">The package id.</param>
    /// <returns>The install, or null when the listing has no such package.</returns>
    public FeedInstallViewModel? Update(string id)
    {
        var package = _all.FirstOrDefault(candidate => candidate.Id == id);
        return package is null ? null : Install(package);
    }

    [RelayCommand]
    private void ClearFinishedInstalls()
    {
        foreach (var install in Installs.Where(install => !install.IsActive && !install.CanRetry).ToList())
        {
            Installs.Remove(install);
        }

        NotifyFinishedChanged();
    }

    private void NotifyFinishedChanged()
    {
        OnPropertyChanged(nameof(FinishedInstalls));
        OnPropertyChanged(nameof(HasFinishedInstalls));
        OnPropertyChanged(nameof(FinishedSummary));
    }

    private bool CanInstall() => Selected is { IsInstallable: true };

    [RelayCommand(CanExecute = nameof(CanInstall))]
    private void InstallSelected()
    {
        if (Selected is { } package)
        {
            Install(package);
        }
    }

    partial void OnSelectedKindChanged(FilterOption<string?> value) => ApplyFilter();

    partial void OnQueryChanged(string value) => ApplyFilter();

    partial void OnOnlyFittingChanged(bool value) => ApplyFilter();

    private void OnInstalled(FeedInstallViewModel install, FeedInstallResult result)
    {
        OnPropertyChanged(nameof(ActiveInstalls));
        Installed?.Invoke(this, EventArgs.Empty);
    }

    private void OnSettingsChanged()
    {
        string key = $"{_services.Feed.Url}#{_services.Feed.Channel}";
        if (_load is not null && key != _loadedKey)
        {
            _ = RefreshAsync();
        }
    }

    private async Task LoadCoreAsync()
    {
        int generation = ++_generation;
        string key = $"{_services.Feed.Url}#{_services.Feed.Channel}";
        _loadedKey = key;
        IsLoading = true;
        Status = "Reading the Sharplight feed…";
        try
        {
            var listing = await Task.Run(() => _services.Feed.GetListingAsync(CancellationToken.None));
            if (generation != _generation)
            {
                return;
            }

            var packages = await Task.Run(() => listing.Packages.Select(summary => new FeedPackageViewModel(summary, _services.Feed.Environment, listing.Verified)).ToList());
            _all.Clear();
            _all.AddRange(packages);
            Kinds.Clear();
            Kinds.Add(s_allKinds);
            foreach (var group in _all.GroupBy(package => package.Kind).OrderBy(group => KindOrder(group.Key)))
            {
                Kinds.Add(new FilterOption<string?>($"{ModelKindCatalog.For(group.Key).Title} ({group.Count()})", group.Key));
            }

            SelectedKind = s_allKinds;
            Failure = listing.Failure;
            IsOffline = listing.FromCache;
            IsListingVerified = listing.Verified;
            ListingText = listing.Failure is null
                ? $"Sharplight feed · {_services.Feed.Channel} · {ByteFormat.Count(_all.Count, "package", "packages")}"
                : $"Sharplight feed · {_services.Feed.Channel}";
            OfflineText = listing.FromCache
                ? $"The feed could not be reached. Showing the last verified listing{(listing.Updated is { } at ? $" from {at.LocalDateTime:yyyy-MM-dd HH:mm}" : "")}; installing needs the network."
                : "";
            if (listing.Failure is null)
            {
                await EstimateFitAsync();
            }

            RefreshInstalled();
            Status = listing.Failure is null && _all.Count == 0 ? "The feed has no packages in this channel yet." : "";
            ApplyFilter();
            ListingChanged?.Invoke(this, EventArgs.Empty);
        }
        finally
        {
            if (generation == _generation)
            {
                IsLoading = false;
            }
        }
    }

    private async Task EstimateFitAsync()
    {
        try
        {
            var memory = await _advisor.GetMemoryAsync(CancellationToken.None);
            long gpu = memory.Where(device => device.IsGpu).Sum(device => device.TotalBytes);
            long system = memory.Where(device => !device.IsGpu).Select(device => device.TotalBytes).DefaultIfEmpty(0).Max();
            foreach (var package in _all)
            {
                package.EstimateFit(gpu, system);
            }
        }
        catch (Exception error) when (error is not OutOfMemoryException)
        {
            // Without device figures the packages show no fit badge.
        }
    }

    private void ApplyFilter()
    {
        var selected = Selected;
        var kind = SelectedKind?.Value;
        var words = Query.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        Entries.Clear();
        foreach (var package in _all)
        {
            bool matches = (kind is null || package.Kind == kind)
                && words.All(word => package.SearchText.Contains(word, StringComparison.OrdinalIgnoreCase))
                && (!OnlyFitting || package.Fit is CatalogFit.Gpu or CatalogFit.Cpu);
            if (matches)
            {
                Entries.Add(package);
            }
        }

        Selected = selected is not null && Entries.Contains(selected) ? selected : Entries.FirstOrDefault();
        if (_all.Count > 0 && Entries.Count == 0)
        {
            Status = "No package matches.";
        }
        else if (Status == "No package matches.")
        {
            Status = "";
        }
    }

    private static int KindOrder(string kind)
    {
        int index = ModelKindCatalog.All.ToList().FindIndex(info => info.Id == kind);
        return index < 0 ? int.MaxValue : index;
    }
}
