using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Package;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services;
using Sharplight.Studio.Services.Library;
using Sharplight.Studio.Services.Platform;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.ViewModels.Inspector;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>The local library: loaded models, stored models and added files.</summary>
internal sealed partial class LibraryViewModel : ObservableObject
{
    private static readonly FileFilter s_modelFiles = new("Models", ["*.slm", "*.gguf", "*.onnx", "*.bin"]);
    private static readonly FileFilter s_packageFiles = new("Sharplight packages", ["*.slm"]);

    private readonly StudioServices _services;
    private readonly INavigator _navigator;

    public LibraryViewModel(StudioServices services, INavigator navigator)
    {
        _services = services;
        _navigator = navigator;
        services.Runtime.ModelsChanged += (_, _) => services.Dispatcher.Post(MarkLoaded);
        services.Settings.Changed += (_, _) => services.Dispatcher.Post(() => OnPropertyChanged(nameof(StoreRoot)));
    }

    /// <summary>The model feed whose listing says which installed packages have a newer version; null until the page wires it.</summary>
    public FeedCatalogViewModel? Feed { get; set; }

    /// <summary>Stored models and added files.</summary>
    public ObservableCollection<LibraryEntryViewModel> Entries { get; } = [];

    /// <summary>The entries the kind filter lets through (all of <see cref="Entries"/> for "All kinds").</summary>
    public ObservableCollection<LibraryEntryViewModel> VisibleEntries { get; } = [];

    /// <summary>The kind filter choices: all kinds, then every kind present in the library with its count.</summary>
    public ObservableCollection<KindFilterViewModel> KindFilters { get; } = [];

    /// <summary>The chosen kind filter.</summary>
    [ObservableProperty]
    public partial KindFilterViewModel? SelectedKindFilter { get; set; }

    /// <summary>More than one kind is in the library, so the filter is worth showing.</summary>
    public bool HasKindFilter => KindFilters.Count > 2;

    /// <summary>The library has models, but none of the chosen kind.</summary>
    public bool IsFilteredEmpty => Entries.Count > 0 && VisibleEntries.Count == 0;

    partial void OnSelectedKindFilterChanged(KindFilterViewModel? value) => ApplyKindFilter();

    /// <summary>The store directory.</summary>
    public string StoreRoot => _services.Settings.StoreRoot;

    /// <summary>Why the store is where it is.</summary>
    public string StoreSource => _services.Settings.StoreRootSource;

    /// <summary>The open load panel, or null.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsPanelOpen))]
    public partial LoadOptionsViewModel? LoadPanel { get; set; }

    /// <summary>The open convert panel, or null.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsPanelOpen))]
    public partial ConvertViewModel? ConvertPanel { get; set; }

    /// <summary>The open package inspector (shown instead of the models page), or null.</summary>
    [ObservableProperty]
    public partial PackageInspectorViewModel? Inspector { get; set; }

    /// <summary>Listing.</summary>
    [ObservableProperty]
    public partial bool IsRefreshing { get; set; }

    /// <summary>Status or error text.</summary>
    [ObservableProperty]
    public partial string Status { get; set; } = "";

    /// <summary>A side panel is open.</summary>
    public bool IsPanelOpen => LoadPanel is not null || ConvertPanel is not null;

    /// <summary>True when no model is on disk.</summary>
    public bool IsEmpty => Entries.Count == 0;


    /// <summary>The manifests of the stored (pulled or installed) models.</summary>
    public IReadOnlyList<Sharplight.Hub.ModelManifest> StoredManifests => [.. Entries.Select(entry => entry.Entry.Manifest).OfType<Sharplight.Hub.ModelManifest>()];

    /// <summary>Names (<c>owner/name:tag</c>) of the stored models.</summary>
    public IReadOnlySet<string> StoredNames => Entries.Where(entry => entry.Entry.Manifest is not null)
        .Select(entry => entry.Entry.Manifest!.Name).ToHashSet(StringComparer.OrdinalIgnoreCase);

    /// <summary>Lists the models again.</summary>
    [RelayCommand]
    public async Task RefreshAsync()
    {
        IsRefreshing = true;
        try
        {
            var entries = await _services.Library.ListAsync(CancellationToken.None);
            var loadedPaths = _services.Runtime.LoadedModels.Select(model => model.Path).ToHashSet();
            Entries.Clear();
            foreach (var entry in entries)
            {
                Entries.Add(new LibraryEntryViewModel(entry) { IsLoaded = loadedPaths.Contains(entry.Path), Library = this });
            }

            RebuildKindFilters();
            ApplyFeedUpdates();

            Status = Entries.Count == 1 ? "1 model" : $"{Entries.Count} models";
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            Status = error.Message;
        }
        finally
        {
            IsRefreshing = false;
            OnPropertyChanged(nameof(IsEmpty));
            OnPropertyChanged(nameof(StoreRoot));
            OnPropertyChanged(nameof(StoreSource));
        }
    }

    /// <summary>Marks the entries that came from the model feed and have a newer version listed there.</summary>
    public void ApplyFeedUpdates()
    {
        foreach (var entry in Entries)
        {
            entry.FeedUpdateVersion = entry.Entry.Manifest is { } manifest ? Feed?.NewerVersionOf(manifest) : null;
        }
    }

    /// <summary>Installs the newer feed version of <paramref name="entry"/>; the old package stays until the new one verified.</summary>
    [RelayCommand]
    private void UpdateFromFeed(LibraryEntryViewModel entry)
    {
        if (Feed is null || entry.Entry.Manifest?.FeedPackage is not { } id)
        {
            return;
        }

        var install = Feed.Update(id);
        if (install is null)
        {
            Status = $"{entry.Name} is no longer in the feed.";
            return;
        }

        entry.IsUpdating = true;
        Status = $"Updating {entry.Name} to {install.Version}…";
        _ = FinishUpdateAsync(entry, install);
    }

    private async Task FinishUpdateAsync(LibraryEntryViewModel entry, FeedInstallViewModel install)
    {
        await install.Completion;
        entry.IsUpdating = false;
        Status = install.Error ?? (install.IsDone ? $"{entry.Name} updated to {install.Version}." : "Update cancelled.");
    }

    /// <summary>Rebuilds the kind choices from the entries; the chosen kind stays chosen while the library still has models of it.</summary>
    private void RebuildKindFilters()
    {
        string? chosen = SelectedKindFilter?.Kind;
        KindFilters.Clear();
        KindFilters.Add(new KindFilterViewModel(null, Entries.Count));
        foreach (var info in ModelKindCatalog.All.Concat(Entries.Select(entry => entry.KindInfo)).DistinctBy(info => info.Id, StringComparer.OrdinalIgnoreCase))
        {
            int count = Entries.Count(entry => string.Equals(entry.KindInfo.Id, info.Id, StringComparison.OrdinalIgnoreCase));
            if (count > 0)
            {
                KindFilters.Add(new KindFilterViewModel(info, count));
            }
        }

        OnPropertyChanged(nameof(HasKindFilter));
        SelectedKindFilter = KindFilters.FirstOrDefault(filter => filter.Kind == chosen) ?? KindFilters[0];
        ApplyKindFilter();
    }

    /// <summary>Shows only the models of <paramref name="kind"/> (all kinds when the library has none of it).</summary>
    /// <param name="kind">The kind (<see cref="Sharplight.Models.ModelKinds"/>).</param>
    public void SelectKind(string kind)
    {
        SelectedKindFilter = KindFilters.FirstOrDefault(filter => string.Equals(filter.Kind, kind, StringComparison.OrdinalIgnoreCase)) ?? KindFilters.FirstOrDefault();
    }

    private void ApplyKindFilter()
    {
        string? kind = SelectedKindFilter?.Kind;
        var shown = Entries.Where(entry => kind is null || string.Equals(entry.KindInfo.Id, kind, StringComparison.OrdinalIgnoreCase)).ToList();
        if (!shown.SequenceEqual(VisibleEntries))
        {
            VisibleEntries.Clear();
            foreach (var entry in shown)
            {
                VisibleEntries.Add(entry);
            }
        }

        OnPropertyChanged(nameof(IsFilteredEmpty));
    }

    [RelayCommand]
    private void Load(LibraryEntryViewModel entry)
    {
        if (entry.IsPrecompiling)
        {
            Status = $"{entry.Id} is being precompiled; load it when that finished.";
            return;
        }

        if (entry.Entry.Kind == ModelKindCatalog.Lora)
        {
            Status = $"{entry.Name} is a LoRA adapter: add it in the load panel of an image model.";
            return;
        }

        if (!entry.IsRunnable)
        {
            Status = $"{entry.Name} is a {entry.KindTitle.ToLowerInvariant()} model: Sharplight cannot run models of this kind yet.";
            return;
        }

        ConvertPanel = null;
        LoadPanel = new LoadOptionsViewModel(entry, _services, OnLoaded, () => LoadPanel = null,
            () => [.. Entries.Select(other => other.Entry)], OpenVariant);
    }

    /// <summary>Switches the load panel to another library model (a smaller quantization the estimate suggests).</summary>
    private void OpenVariant(LibraryEntry variant)
    {
        if (Entries.FirstOrDefault(other => other.Entry.Path == variant.Path) is { } entry)
        {
            LoadPanel?.CloseCommand.Execute(null);
            Load(entry);
        }
    }


    /// <summary>Opens the package inspector for a library model.</summary>
    [RelayCommand]
    public Task InspectAsync(LibraryEntryViewModel entry) => OpenInspectorAsync(entry.Entry.Path, entry.Entry.Manifest);

    /// <summary>Opens the package inspector for any <c>.slm</c> file the user picks.</summary>
    [RelayCommand]
    private async Task OpenPackageFileAsync()
    {
        var files = await _services.Platform.PickFilesAsync("Open .slm package", [s_packageFiles], allowMultiple: false);
        if (files.Count > 0)
        {
            await OpenInspectorAsync(files[0], null);
        }
    }

    /// <summary>Opens the package inspector for <paramref name="path"/> and reads the package.</summary>
    /// <returns>Whether the package was read.</returns>
    public Task<bool> OpenInspectorAsync(string path, Sharplight.Hub.ModelManifest? manifest)
    {
        var inspector = new PackageInspectorViewModel(path, manifest, () => Inspector = null);
        Inspector = inspector;
        return inspector.LoadAsync();
    }

    [RelayCommand]
    private void OpenEntryFolder(LibraryEntryViewModel entry)
    {
        var folder = Directory.Exists(entry.Entry.Path) ? entry.Entry.Path : Path.GetDirectoryName(entry.Entry.Path);
        if (folder is not null && Directory.Exists(folder))
        {
            _services.Platform.Open(folder);
        }
    }

    [RelayCommand]
    private async Task DeleteAsync(LibraryEntryViewModel entry)
    {
        if (!entry.ConfirmDelete)
        {
            foreach (var other in Entries)
            {
                other.ConfirmDelete = false;
            }

            entry.ConfirmDelete = true;
            return;
        }

        if (entry.IsLoaded || entry.IsPrecompiling)
        {
            Status = entry.IsLoaded ? $"Unload {entry.Id} before deleting it." : $"{entry.Id} is being precompiled; cancel that first.";
            entry.ConfirmDelete = false;
            return;
        }

        try
        {
            await _services.Library.RemoveAsync(entry.Entry);
            Entries.Remove(entry);
            RebuildKindFilters();
            Status = entry.IsLocalFile ? $"Removed {entry.Name} from the list (the file was kept)." : $"Deleted {entry.Name}.";
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or Sharplight.Hub.HubException)
        {
            Status = error.Message;
        }

        OnPropertyChanged(nameof(IsEmpty));
    }

    /// <summary>
    /// Precompiles a library package for every execution provider and device on this machine and writes the code into the package in
    /// place. Refused while the model is resident (the runtime maps the file); the entry and an open inspector of it are read again
    /// afterwards.
    /// </summary>
    [RelayCommand]
    public async Task PrecompileAsync(LibraryEntryViewModel entry)
    {
        var path = entry.Entry.Path;
        if (!entry.IsPackage)
        {
            Status = entry.PrecompileHint;
            return;
        }

        if (entry.IsLoaded || _services.Runtime.LoadedModels.Any(model => model.Path == path))
        {
            entry.IsLoaded = true;
            Status = $"Unload {entry.Id} before precompiling it: the package is rewritten in place.";
            return;
        }

        if (entry.IsPrecompiling || _services.Precompiler.IsBusy(path))
        {
            return;
        }

        using var cancellation = new CancellationTokenSource();
        entry.PrecompileCancellation = cancellation;
        entry.BeginPrecompile();
        Status = $"Precompiling {entry.Name}…";
        using var progress = new DispatchingProgress<PrecompileProgress>(_services.Dispatcher, entry.OnPrecompileProgress, SameStep);
        try
        {
            var report = await Task.Run(() => _services.Precompiler.PrecompileAsync(path, progress, cancellation.Token), CancellationToken.None);
            progress.Flush();
            entry.CompletePrecompile(report);
            Status = $"{entry.Name}: {entry.PrecompileStatus}";
            await ReloadAsync(entry);
        }
        catch (OperationCanceledException)
        {
            entry.PrecompileStatus = "Cancelled; the package was not changed.";
            Status = $"Precompiling {entry.Name} was cancelled.";
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or InvalidOperationException or InvalidDataException
            or NotSupportedException or MissingSymbolBoundException)
        {
            entry.PrecompileStatus = $"Failed: {error.Message} The package was not changed.";
            Status = $"Precompiling {entry.Name} failed.";
        }
        finally
        {
            progress.Flush();
            entry.PrecompileCancellation = null;
            entry.IsPrecompiling = false;
        }
    }

    /// <summary>A kernel count of the same device and stage supersedes the one before it (only the latest is shown).</summary>
    private static bool SameStep(PrecompileProgress previous, PrecompileProgress next)
        => previous.Provider == next.Provider && previous.Device == next.Device && previous.Stage == next.Stage
            && previous.Kernels is not null && next.Kernels is not null;

    /// <summary>Cancels a running precompile; nothing is written then.</summary>
    [RelayCommand]
    private static void CancelPrecompile(LibraryEntryViewModel entry) => entry.PrecompileCancellation?.Cancel();

    /// <summary>Reads the entry again (size, precompiled sets) and refreshes an open inspector of the same package.</summary>
    private async Task ReloadAsync(LibraryEntryViewModel entry)
    {
        try
        {
            entry.Entry = await _services.Library.ReloadAsync(entry.Entry, CancellationToken.None);
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            Status = error.Message;
        }

        if (Inspector is { } inspector && inspector.Path == entry.Entry.Path)
        {
            await inspector.LoadAsync();
        }
    }

    [RelayCommand]
    private async Task AddFileAsync()
    {
        var files = await _services.Platform.PickFilesAsync("Add models", [s_modelFiles], allowMultiple: true);
        foreach (var file in files)
        {
            await _services.Library.AddLocalAsync(file);
        }

        if (files.Count > 0)
        {
            await RefreshAsync();
        }
    }

    [RelayCommand]
    private void OpenConvert()
    {
        LoadPanel = null;
        ConvertPanel = new ConvertViewModel(_services, () => _ = RefreshAsync(), () => ConvertPanel = null);
    }

    [RelayCommand]
    private void OpenStoreFolder()
    {
        Directory.CreateDirectory(StoreRoot);
        _services.Platform.Open(StoreRoot);
    }

    [RelayCommand]
    private void OpenDiscover() => _navigator.OpenDiscover();

    private void OnLoaded(ILoadedModel model)
    {
        LoadPanel = null;
        Status = $"Loaded {model.Id} on {model.Device.Name}.";
        if (!_navigator.OpenKind(model.Kind, model))
        {
            _navigator.OpenChat(model);
        }
    }

    private void MarkLoaded()
    {
        var paths = _services.Runtime.LoadedModels.Select(model => model.Path).ToHashSet();
        foreach (var entry in Entries)
        {
            entry.IsLoaded = paths.Contains(entry.Entry.Path);
        }
    }
}
