using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Studio.Services;
using Sharplight.Studio.Services.Library;
using Sharplight.Studio.Services.Platform;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.ViewModels.Inspector;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>The local library: loaded models, stored models and added files.</summary>
internal sealed partial class LibraryViewModel : ObservableObject
{
    private static readonly FileFilter s_modelFiles = new("Models", ["*.slm", "*.gguf", "*.onnx"]);
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

    /// <summary>Stored models and added files.</summary>
    public ObservableCollection<LibraryEntryViewModel> Entries { get; } = [];

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

    [RelayCommand]
    private void Load(LibraryEntryViewModel entry)
    {
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

        if (entry.IsLoaded)
        {
            Status = $"Unload {entry.Id} before deleting it.";
            entry.ConfirmDelete = false;
            return;
        }

        try
        {
            await _services.Library.RemoveAsync(entry.Entry);
            Entries.Remove(entry);
            Status = entry.IsLocalFile ? $"Removed {entry.Name} from the list (the file was kept)." : $"Deleted {entry.Name}.";
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or Sharplight.Hub.HubException)
        {
            Status = error.Message;
        }

        OnPropertyChanged(nameof(IsEmpty));
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
        _navigator.OpenChat(model);
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
