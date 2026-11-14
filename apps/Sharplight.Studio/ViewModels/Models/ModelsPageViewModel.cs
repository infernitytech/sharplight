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

    private bool _loaded;

    public ModelsPageViewModel(StudioServices services, INavigator navigator)
    {
        Downloads = new DownloadsViewModel(services);
        Library = new LibraryViewModel(services, navigator);
        Resident = new ResidentModelsViewModel(services, navigator);
        Discover = new HubSearchViewModel(services, Downloads, () => Library.StoredNames);
        Downloads.ModelStored += (_, _) => _ = Library.RefreshAsync();
        Downloads.Enqueued += (_, _) => SelectedTab = DownloadsTab;
        Downloads.PropertyChanged += (_, args) =>
        {
            if (args.PropertyName == nameof(DownloadsViewModel.ActiveCount))
            {
                Badge = Downloads.ActiveCount;
            }
        };
        Downloads.Restore();
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

    /// <summary>The visible tab.</summary>
    [ObservableProperty]
    public partial int SelectedTab { get; set; } = LibraryTab;

    /// <inheritdoc/>
    public override async Task ActivateAsync()
    {
        if (!_loaded)
        {
            _loaded = true;
            await Library.RefreshAsync();
            if (Library.IsEmpty)
            {
                SelectedTab = DiscoverTab;
            }
        }
    }
}
