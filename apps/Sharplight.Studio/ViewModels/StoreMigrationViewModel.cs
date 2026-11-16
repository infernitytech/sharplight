using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Hub;
using Sharplight.Studio.Services;

namespace Sharplight.Studio.ViewModels;

/// <summary>
/// The one-time offer shown when the model store in effect is empty while another store (the system store, or the folder next to the app of earlier versions)
/// has models: keep using that store, or move or copy its models into the store in effect.
/// </summary>
internal sealed partial class StoreMigrationViewModel : ObservableObject
{
    private readonly StudioServices _services;
    private readonly Action _storeChanged;

    public StoreMigrationViewModel(StudioServices services, Action storeChanged)
    {
        _services = services;
        _storeChanged = storeChanged;
    }

    /// <summary>The offer is shown.</summary>
    [ObservableProperty]
    public partial bool IsOpen { get; set; }

    /// <summary>The store that has the models.</summary>
    [ObservableProperty]
    public partial string PreviousRoot { get; set; } = "";

    /// <summary>The store in effect (empty).</summary>
    [ObservableProperty]
    public partial string StoreRoot { get; set; } = "";

    /// <summary>Models in <see cref="PreviousRoot"/>.</summary>
    [ObservableProperty]
    public partial int ModelCount { get; set; }

    /// <summary>Moving or copying.</summary>
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(MoveCommand), nameof(CopyCommand), nameof(KeepPreviousCommand))]
    public partial bool IsBusy { get; set; }

    /// <summary>The choice was carried out; only closing is left.</summary>
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(MoveCommand), nameof(CopyCommand), nameof(KeepPreviousCommand))]
    [NotifyPropertyChangedFor(nameof(DismissText))]
    public partial bool IsDone { get; set; }

    /// <summary>Text of the button that closes the offer.</summary>
    public string DismissText => IsDone ? "Close" : "Keep the new folder";

    /// <summary>The offer text, or the result.</summary>
    [ObservableProperty]
    public partial string Message { get; set; } = "";

    /// <summary>Shows the offer when it was not answered yet and another store has models while the store in effect has none.</summary>
    public void Check()
    {
        if (_services.Settings.Current.StoreMigrationAnswered || _services.Settings.FindPreviousStore() is not { } previous)
        {
            return;
        }

        PreviousRoot = previous;
        StoreRoot = _services.Settings.StoreRoot;
        ModelCount = new ModelStore(previous).List().Count;
        Message = $"{ModelCount} model(s) are in {PreviousRoot}, but the model store is now {StoreRoot}. "
            + "Keep using the old folder, or move or copy the models to the new one.";
        IsDone = false;
        IsOpen = true;
    }

    private bool CanChoose() => !IsBusy && !IsDone;

    /// <summary>Keeps using the previous store: the system folder, or that folder as a custom location.</summary>
    [RelayCommand(CanExecute = nameof(CanChoose))]
    private void KeepPrevious()
    {
        bool isUserStore = string.Equals(new ModelStore(PreviousRoot).Root, new ModelStore(ModelStore.SystemRoot).Root, StringComparison.Ordinal);
        _services.Settings.Update(current => isUserStore
            ? current with { ModelsLocation = ModelStoreLocation.User, StoreMigrationAnswered = true }
            : current with { ModelsLocation = ModelStoreLocation.Custom, ModelsDirectory = PreviousRoot, StoreMigrationAnswered = true });
        IsOpen = false;
        _storeChanged();
    }

    /// <summary>Moves the models into the store in effect.</summary>
    [RelayCommand(CanExecute = nameof(CanChoose))]
    private Task MoveAsync() => TransferAsync(move: true);

    /// <summary>Copies the models into the store in effect.</summary>
    [RelayCommand(CanExecute = nameof(CanChoose))]
    private Task CopyAsync() => TransferAsync(move: false);

    /// <summary>Closes the offer for good, leaving the stores as they are.</summary>
    [RelayCommand]
    private void Dismiss()
    {
        if (!_services.Settings.Current.StoreMigrationAnswered)
        {
            _services.Settings.Update(current => current with { StoreMigrationAnswered = true });
        }

        IsOpen = false;
    }

    private async Task TransferAsync(bool move)
    {
        IsBusy = true;
        Message = move ? $"Moving the models to {StoreRoot}…" : $"Copying the models to {StoreRoot}…";
        try
        {
            var source = new ModelStore(PreviousRoot);
            var destination = new ModelStore(StoreRoot);
            int count = await Task.Run(() => ModelStoreTransfer.Transfer(source, destination, move));
            _services.Settings.Update(current => current with { StoreMigrationAnswered = true });
            Message = $"{count} model(s) {(move ? "moved" : "copied")} to {StoreRoot}.";
            IsDone = true;
            _storeChanged();
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or HubException)
        {
            Message = $"Could not {(move ? "move" : "copy")} the models: {error.Message}";
        }
        finally
        {
            IsBusy = false;
        }
    }
}
