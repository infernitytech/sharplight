using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Hosting.Fit;
using Sharplight.Hub;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services;
using Sharplight.Studio.Services.Fit;
using Sharplight.Studio.Services.Library;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>
/// Discover's "Verified" list: the models of the verified catalog (docs/verified-catalog.md) per task, with their variants, sizes, the
/// fit on this machine, licenses and verification, installed with their companions in one click.
/// </summary>
internal sealed partial class VerifiedCatalogViewModel : ObservableObject
{
    private static readonly FilterOption<string?> s_allKinds = new("All tasks", null);

    private readonly StudioServices _services;
    private readonly DownloadsViewModel _downloads;
    private readonly Func<IReadOnlyList<ModelManifest>> _storedModels;
    private readonly FitAdvisor _advisor;
    private readonly List<CatalogEntryViewModel> _all = [];
    private Task? _load;
    private string? _pendingKind;

    public VerifiedCatalogViewModel(StudioServices services, DownloadsViewModel downloads, Func<IReadOnlyList<ModelManifest>> storedModels)
    {
        _services = services;
        _downloads = downloads;
        _storedModels = storedModels;
        _advisor = FitAdvisor.For(services);
        Kinds.Add(s_allKinds);
        SelectedKind = s_allKinds;
        // Memory moved (models loaded or unloaded, other applications): rate the variants again.
        services.Memory.Changed += (_, memory) => services.Dispatcher.Post(() =>
        {
            if (_all.Count > 0)
            {
                RateFit(memory);
                ApplyFilter();
            }
        });
    }

    /// <summary>Raised on the UI thread when the catalog has been loaded.</summary>
    public event EventHandler? Loaded;

    /// <summary>The tasks the catalog has models for, with their counts, and "All tasks".</summary>
    public ObservableCollection<FilterOption<string?>> Kinds { get; } = [];

    [ObservableProperty]
    public partial FilterOption<string?> SelectedKind { get; set; }

    [ObservableProperty]
    public partial string Query { get; set; } = "";

    /// <summary>Only models with a variant that fits this machine's memory.</summary>
    [ObservableProperty]
    public partial bool OnlyFitting { get; set; }

    /// <summary>The entries that match the task, the search text and the fit filter.</summary>
    public ObservableCollection<CatalogEntryViewModel> Entries { get; } = [];

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasSelection), nameof(SelectedVariants))]
    public partial CatalogEntryViewModel? SelectedEntry { get; set; }

    /// <summary>The variants of the selected entry (empty without a selection).</summary>
    public IReadOnlyList<CatalogVariantViewModel> SelectedVariants => SelectedEntry?.Variants ?? [];

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(InstallCommand))]
    [NotifyPropertyChangedFor(nameof(InstallText), nameof(SelectedCompanions), nameof(HasSelectedCompanions))]
    public partial CatalogVariantViewModel? SelectedVariant { get; set; }

    /// <summary>The companions the selected variant installs (label and size).</summary>
    public IReadOnlyList<string> SelectedCompanions => SelectedVariant?.Companions ?? [];

    public bool HasSelectedCompanions => SelectedCompanions.Count > 0;

    public bool HasSelection => SelectedEntry is not null;

    [ObservableProperty]
    public partial bool IsLoading { get; set; }

    [ObservableProperty]
    public partial string Status { get; set; } = "";

    /// <summary>The catalog's date and size, e.g. <c>Verified catalog of 2026-11-16 · 157 models</c>.</summary>
    [ObservableProperty]
    public partial string CatalogText { get; set; } = "";

    /// <summary>The installable entries of the catalog.</summary>
    public IReadOnlyList<CatalogEntryViewModel> AllEntries => _all;

    public string InstallText => SelectedVariant is { IsInstalled: true } ? "Installed · install again" : "Download and install";

    /// <summary>Loads the catalog and the machine's memory once (both off the UI thread).</summary>
    public Task LoadAsync() => _load ??= LoadCoreAsync();

    /// <summary>Whether the catalog has installable models of the kind of <paramref name="task"/> (any model when null).</summary>
    public bool HasEntriesFor(ModelTask? task)
    {
        var kind = KindOf(task);
        return _all.Any(entry => kind is null || entry.Entry.Kind == kind);
    }

    /// <summary>Shows the models of <paramref name="task"/>'s kind (every model when null or when the catalog has none of that kind).</summary>
    public void SelectTask(ModelTask? task)
    {
        var kind = KindOf(task);
        if (_all.Count == 0)
        {
            _pendingKind = kind;
            return;
        }

        SelectedKind = Kinds.FirstOrDefault(option => option.Value == kind) ?? s_allKinds;
    }

    /// <summary>Shows the models of <paramref name="kind"/> (every model when the catalog has none of it).</summary>
    /// <param name="kind">The catalog kind.</param>
    public void SelectKind(string kind)
    {
        if (_all.Count == 0)
        {
            _pendingKind = kind;
            return;
        }

        SelectedKind = Kinds.FirstOrDefault(option => option.Value == kind) ?? s_allKinds;
    }

    /// <summary>Marks the variants that are in the library (after a download or a library change).</summary>
    public void RefreshInstalled()
    {
        var stored = _storedModels();
        foreach (var entry in _all)
        {
            foreach (var variant in entry.Variants)
            {
                variant.IsInstalled = stored.Any(variant.Plan.IsInstalledAs);
            }

            entry.IsInstalled = entry.Variants.Any(variant => variant.IsInstalled);
        }

        OnPropertyChanged(nameof(InstallText));
    }

    partial void OnSelectedKindChanged(FilterOption<string?> value) => ApplyFilter();

    partial void OnQueryChanged(string value) => ApplyFilter();

    partial void OnOnlyFittingChanged(bool value) => ApplyFilter();

    partial void OnSelectedEntryChanged(CatalogEntryViewModel? value)
    {
        SelectedVariant = value is null ? null
            : value.Variants.FirstOrDefault(variant => variant.Recommended && variant.Fit != CatalogFit.TooLarge)
              ?? value.Variants.FirstOrDefault(variant => variant.Fit is CatalogFit.Gpu or CatalogFit.Cpu)
              ?? value.Variants[0];
    }

    private bool CanInstall() => SelectedEntry is not null && SelectedVariant is not null;

    [RelayCommand(CanExecute = nameof(CanInstall))]
    private async Task InstallAsync()
    {
        var plan = SelectedVariant!.Plan;
        if (!await _services.Licenses.RequestAsync(LicenseNotices.TermsFor(plan.Entry, plan.Variant)))
        {
            return;
        }

        _downloads.Enqueue(plan.Reference, plan.TotalBytes, plan.Entry.Kind, plan.Key, $"{plan.Entry.Name} · {plan.Variant.Label}");
    }

    [RelayCommand]
    private void OpenOnHub()
    {
        if (SelectedVariant is { } variant)
        {
            _services.Platform.Open(variant.Variant.GetPageUri(HubClientOptions.DefaultEndpoint).ToString());
        }
    }

    private async Task LoadCoreAsync()
    {
        IsLoading = true;
        Status = "Loading the verified catalog…";
        try
        {
            var catalog = await Task.Run(() => _services.Hub.GetCatalogAsync(CancellationToken.None));
            var entries = await Task.Run(() => catalog.Entries.Where(entry => entry.IsInstallable).Select(entry => new CatalogEntryViewModel(entry)).ToList());
            _all.Clear();
            _all.AddRange(entries);
            Kinds.Clear();
            Kinds.Add(s_allKinds);
            foreach (var group in _all.GroupBy(entry => entry.Entry.Kind).OrderBy(group => KindOrder(group.Key)))
            {
                Kinds.Add(new FilterOption<string?>($"{ModelKindCatalog.For(group.Key).Title} ({group.Count()})", group.Key));
            }

            CatalogText = $"Verified catalog of {catalog.Updated} · {ByteFormat.Count(_all.Count, "model", "models")}";
            await EstimateFitAsync();
            RefreshInstalled();
            Status = "";
            SelectedKind = Kinds.FirstOrDefault(option => option.Value == _pendingKind) ?? s_allKinds;
            ApplyFilter();
            Loaded?.Invoke(this, EventArgs.Empty);
        }
        catch (Exception error) when (error is InvalidDataException or InvalidOperationException or IOException)
        {
            Status = $"The verified catalog could not be read: {error.Message}";
        }
        finally
        {
            IsLoading = false;
        }
    }

    private async Task EstimateFitAsync()
    {
        try
        {
            RateFit(await _advisor.GetMemoryAsync(CancellationToken.None));
        }
        catch (Exception error) when (error is not OutOfMemoryException)
        {
            // Without device figures the variants show no fit badge.
        }
    }

    private void RateFit(IReadOnlyList<DeviceMemory> memory)
    {
        // What the devices can hold for Sharplight (memory other applications occupy is not counted), not their total memory.
        long gpu = memory.Where(device => device.IsGpu).Sum(device => device.UsableBytes);
        long system = memory.Where(device => !device.IsGpu).Select(device => device.UsableBytes).DefaultIfEmpty(0).Max();
        foreach (var variant in _all.SelectMany(entry => entry.Variants))
        {
            variant.EstimateFit(gpu, system);
        }
    }

    private void ApplyFilter()
    {
        var selected = SelectedEntry;
        var kind = SelectedKind?.Value;
        var words = Query.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        Entries.Clear();
        foreach (var entry in _all)
        {
            bool matches = (kind is null || entry.Entry.Kind == kind)
                && words.All(word => entry.SearchText.Contains(word, StringComparison.OrdinalIgnoreCase))
                && (!OnlyFitting || entry.Variants.Any(variant => variant.Fit is CatalogFit.Gpu or CatalogFit.Cpu));
            if (matches)
            {
                Entries.Add(entry);
            }
        }

        SelectedEntry = selected is not null && Entries.Contains(selected) ? selected : Entries.FirstOrDefault();
        Status = _all.Count > 0 && Entries.Count == 0 ? "No verified model matches." : Status;
    }

    /// <summary>The catalog kind of a Hub task (null for every kind).</summary>
    private static string? KindOf(ModelTask? task) => task is { } value ? ModelKindCatalog.For(value).Id : null;

    private static int KindOrder(string kind)
    {
        int index = ModelKindCatalog.All.ToList().FindIndex(info => info.Id == kind);
        return index < 0 ? int.MaxValue : index;
    }
}
