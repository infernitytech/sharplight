using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Runtime;
using Sharplight.Storage;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services;
using Sharplight.Studio.Services.Settings;

namespace Sharplight.Studio.ViewModels.Settings;

/// <summary>
/// The storage card of the settings page: portable, system or custom location for everything Studio and the engine write, the
/// folders per kind of file (now and with the selected location), and changing the location with the existing files (move, copy or
/// start fresh, optionally leaving the models where they are), with progress. The new location is used after a restart.
/// </summary>
internal sealed partial class StorageSettingsViewModel : ObservableObject
{
    private readonly StudioServices _services;
    private CancellationTokenSource? _cancellation;

    public StorageSettingsViewModel(StudioServices services)
    {
        _services = services;
        Reset();
    }

    /// <summary>The mode choices.</summary>
    public IReadOnlyList<StorageModeChoice> Modes => StorageModeChoice.All;

    /// <summary>The transfer choices.</summary>
    public IReadOnlyList<StorageTransferChoice> Transfers => StorageTransferChoice.All;

    /// <summary>The folders per kind of file.</summary>
    public ObservableCollection<StoragePathRow> Paths { get; } = [];

    /// <summary>The selected mode.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsCustom))]
    public partial StorageModeChoice SelectedMode { get; set; } = StorageModeChoice.All[1];

    /// <summary>The folder of the custom mode.</summary>
    [ObservableProperty]
    public partial string CustomRoot { get; set; } = "";

    /// <summary>What happens to the existing files.</summary>
    [ObservableProperty]
    public partial StorageTransferChoice SelectedTransfer { get; set; } = StorageTransferChoice.All[0];

    /// <summary>Leave the model store where it is.</summary>
    [ObservableProperty]
    public partial bool KeepModels { get; set; }

    /// <summary>The custom mode is selected, so its folder can be chosen.</summary>
    public bool IsCustom => SelectedMode.Value == StorageMode.Custom;

    /// <summary>The location in use and why.</summary>
    public string CurrentDescription => _services.StorageLocation.Current.Description;

    /// <summary>Why the location in use differs from the chosen one (a portable folder that is not writable), or null.</summary>
    public string? Notice => _services.StorageLocation.Current.Notice;

    /// <summary><c>SHARPLIGHT_HOME</c> sets the location: it cannot be changed here.</summary>
    public bool IsLocked => _services.StorageLocation.Current.Source == StorageModeSource.EnvironmentVariable;

    /// <summary>A note on <c>SHARPLIGHT_STUDIO_HOME</c>, or null.</summary>
    public string? StudioHomeNote => _services.StorageLocation.IsStudioHomeOverridden
        ? $"{StudioPaths.HomeVariable} keeps Studio's settings and history in {_services.StorageLocation.Paths.DataDirectory}, whatever the location is."
        : null;

    /// <summary>The selected location differs from the one in use (or the one waiting for the restart).</summary>
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(ApplyCommand))]
    public partial bool HasChange { get; set; }

    /// <summary>The models follow the location, so they can be left behind on their own.</summary>
    [ObservableProperty]
    public partial bool CanKeepModels { get; set; }

    /// <summary>Why the selected change cannot be made, or null.</summary>
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(ApplyCommand))]
    public partial string? Problem { get; set; }

    /// <summary>Files are being transferred.</summary>
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(ApplyCommand), nameof(CancelCommand))]
    public partial bool IsBusy { get; set; }

    /// <summary>Progress of the transfer, 0 to 1.</summary>
    [ObservableProperty]
    public partial double Progress { get; set; }

    /// <summary>Progress of the transfer as text.</summary>
    [ObservableProperty]
    public partial string ProgressText { get; set; } = "";

    /// <summary>Feedback of the last change.</summary>
    [ObservableProperty]
    public partial string? Message { get; set; }

    /// <summary>A new location was recorded: it is used after a restart.</summary>
    [ObservableProperty]
    public partial bool NeedsRestart { get; set; }

    /// <summary>Shows the location in effect again (the selection is reset to it).</summary>
    public void Reset()
    {
        var location = _services.StorageLocation.Pending ?? _services.StorageLocation.Current;
        SelectedMode = StorageModeChoice.Get(location.Mode);
        CustomRoot = location.Mode == StorageMode.Custom ? location.Root ?? "" : CustomRoot;
        Refresh();
    }

    partial void OnSelectedModeChanged(StorageModeChoice value) => Refresh();

    partial void OnCustomRootChanged(string value) => Refresh();

    partial void OnSelectedTransferChanged(StorageTransferChoice value) => Refresh();

    partial void OnKeepModelsChanged(bool value) => Refresh();

    [RelayCommand]
    private async Task BrowseAsync()
    {
        var folder = await _services.Platform.PickFolderAsync("Storage folder");
        if (folder is not null)
        {
            CustomRoot = folder;
            SelectedMode = StorageModeChoice.Get(StorageMode.Custom);
        }
    }

    private bool CanApply() => HasChange && Problem is null && !IsBusy && !IsLocked;

    [RelayCommand(CanExecute = nameof(CanApply))]
    private async Task ApplyAsync()
    {
        var request = Request();
        IsBusy = true;
        Message = null;
        Progress = 0;
        ProgressText = request.Transfer == StorageTransferKind.StartFresh ? "" : "Counting files…";
        _cancellation = new CancellationTokenSource();
        var progress = new Progress<StorageMigrationProgress>(Report);
        try
        {
            var result = await _services.StorageLocation.ChangeAsync(request, progress, _cancellation.Token);
            var verb = request.Transfer == StorageTransferKind.Move ? "moved" : "copied";
            var transferred = request.Transfer == StorageTransferKind.StartFresh
                ? "The existing files were left where they are."
                : $"{result.FilesCopied} file(s) {verb} and verified" + (result.RenamedCategories.Count > 0 ? ", the models moved in one step." : ".");
            var cleanup = request.Transfer == StorageTransferKind.Move ? " The originals are deleted when Studio starts again." : "";
            Message = $"The new location is recorded. {transferred}{cleanup} Restart Studio to use it.";
            NeedsRestart = true;
        }
        catch (OperationCanceledException)
        {
            Message = "Cancelled. The current location stays in use; files copied so far are reused by the next attempt.";
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or InvalidOperationException)
        {
            Message = $"The location was not changed: {error.Message}";
        }
        finally
        {
            _cancellation.Dispose();
            _cancellation = null;
            IsBusy = false;
            Refresh();
        }
    }

    [RelayCommand(CanExecute = nameof(IsBusy))]
    private void Cancel() => _cancellation?.Cancel();

    [RelayCommand]
    private void Restart() => _services.Platform.Restart();

    private void Report(StorageMigrationProgress value)
    {
        Progress = value.Fraction;
        ProgressText = $"{value.FilesDone} of {value.FilesTotal} files, {ByteFormat.Format(value.BytesDone)} of {ByteFormat.Format(value.BytesTotal)}"
            + (value.CurrentFile is null ? "" : $": {value.CurrentFile}");
    }

    private StorageChangeRequest Request() =>
        new(SelectedMode.Value, IsCustom ? CustomRoot : null, SelectedTransfer.Value, KeepModels && CanKeepModels);

    /// <summary>Recomputes the folders and the validity of the selection (also after the model store choice changed).</summary>
    public void Refresh()
    {
        var storage = _services.StorageLocation;
        var inEffect = storage.Pending ?? storage.Current;
        var target = storage.Preview(SelectedMode.Value, IsCustom ? CustomRoot : null);
        HasChange = target is not null && !target.HasSameFolders(inEffect) || target is null && IsCustom;
        CanKeepModels = StorageLocationService.FollowsLayout(_services.Settings.Store);
        Problem = HasChange ? storage.Validate(Request()) : null;
        var next = HasChange ? target : null;
        var studioNext = next is null || storage.IsStudioHomeOverridden ? null : StudioPaths.From(next, null);
        bool modelsMove = CanKeepModels && !KeepModels;
        Paths.Clear();
        Paths.Add(new StoragePathRow("Studio settings", storage.Paths.ConfigDirectory, studioNext?.ConfigDirectory ?? ""));
        Paths.Add(new StoragePathRow("History, presets and outputs", storage.Paths.DataDirectory, studioNext?.DataDirectory ?? ""));
        Paths.Add(new StoragePathRow("Models", _services.Settings.StoreRoot, modelsMove ? next?.ModelsDirectory ?? "" : ""));
        Paths.Add(new StoragePathRow("Compiled kernels", ArtifactStore.DirectoryFor(storage.Current) ?? "off", next is null ? "" : ArtifactStore.DirectoryFor(next) ?? ""));
        Paths.Add(new StoragePathRow("Other caches", storage.Current.CacheDirectory, next?.CacheDirectory ?? ""));
        Paths.Add(new StoragePathRow("Shared tool settings", storage.Current.ConfigDirectory, next?.ConfigDirectory ?? ""));
        Paths.Add(new StoragePathRow("Logs", storage.Current.LogsDirectory, next?.LogsDirectory ?? ""));
    }
}
