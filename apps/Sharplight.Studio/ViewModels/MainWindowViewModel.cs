using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Studio.Services;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.ViewModels.Chat;
using Sharplight.Studio.ViewModels.Image;
using Sharplight.Studio.ViewModels.Inference;
using Sharplight.Studio.ViewModels.Models;
using Sharplight.Studio.ViewModels.Navigation;
using Sharplight.Studio.ViewModels.Search;
using Sharplight.Studio.ViewModels.Server;
using Sharplight.Studio.ViewModels.Settings;
using Sharplight.Studio.ViewModels.Audio;
using Sharplight.Studio.ViewModels.Speech;
using Sharplight.Studio.ViewModels.Stats;
using Sharplight.Studio.ViewModels.Tasks;
using Sharplight.Studio.ViewModels.Updates;
using Sharplight.Studio.ViewModels.Video;
using Sharplight.Studio.ViewModels.Vision;
using Sharplight.Studio.ViewModels.Welcome;

namespace Sharplight.Studio.ViewModels;

/// <summary>The main window: navigation between the pages and the status bar.</summary>
internal sealed partial class MainWindowViewModel : ObservableObject, INavigator
{
    private readonly StudioServices _services;

    public MainWindowViewModel(StudioServices services)
    {
        _services = services;
        Chat = new ChatPageViewModel(services, this);
        Transcribe = new TranscribePageViewModel(services, this);
        Speak = new SpeakPageViewModel(services, this);
        Search = new SearchPageViewModel(services, this);
        Audio = new AudioPageViewModel(services, this);
        Image = new ImagePageViewModel(services, this);
        Video = new VideoPageViewModel(services, this);
        Enhance = new VideoEnhancementPageViewModel(services, this);
        Vision = new VisionPageViewModel(services, this);
        Tasks = new TasksPageViewModel(services, this);
        Models = new ModelsPageViewModel(services, this);
        Welcome = new WelcomePageViewModel(services, this, Models);
        Server = new ServerPageViewModel(services);
        Stats = new StatsPageViewModel(services);
        Settings = new SettingsPageViewModel(services);
        About = new AboutViewModel(services.Platform, services.Updates.CurrentVersion);
        LicenseDialog = new LicenseDialogViewModel(services.Licenses, services.Platform, services.Dispatcher);
        StoreMigration = new StoreMigrationViewModel(services, () => _ = Models.Library.RefreshAsync());
        Navigation = new NavigationViewModel(this, services.Settings);
        Pages = [.. Navigation.Items.Select(item => item.Page).Where(page => !ReferenceEquals(page, Settings)).Distinct()];
        services.Runtime.ModelsChanged += (_, _) => services.Dispatcher.Post(UpdateStatus);
        services.Server.StateChanged += (_, _) => services.Dispatcher.Post(UpdateStatus);
        UpdateStatus();
    }

    /// <summary>Chat page.</summary>
    public ChatPageViewModel Chat { get; }

    /// <summary>Transcribe page: speech to text.</summary>
    public TranscribePageViewModel Transcribe { get; }

    /// <summary>Speak page: text to speech.</summary>
    public SpeakPageViewModel Speak { get; }

    /// <summary>Search page: embeddings over a folder of documents.</summary>
    public SearchPageViewModel Search { get; }

    /// <summary>Audio page: audio codecs and voice activity detection.</summary>
    public AudioPageViewModel Audio { get; }

    /// <summary>Image page: image generation.</summary>
    public ImagePageViewModel Image { get; }

    /// <summary>Video page: video generation.</summary>
    public VideoPageViewModel Video { get; }

    /// <summary>Enhance page: video enhancement (restoring and enlarging videos and images).</summary>
    public VideoEnhancementPageViewModel Enhance { get; }

    /// <summary>Vision page: image classification, embedding, object detection and depth estimation.</summary>
    public VisionPageViewModel Vision { get; }

    /// <summary>Tasks page: the model kinds and where each stands.</summary>
    public TasksPageViewModel Tasks { get; }

    /// <summary>Models page.</summary>
    public ModelsPageViewModel Models { get; }

    /// <summary>The welcome flow: starter models for a new user (not in the navigation).</summary>
    public WelcomePageViewModel Welcome { get; }

    /// <summary>Server page.</summary>
    public ServerPageViewModel Server { get; }

    /// <summary>Statistics page.</summary>
    public StatsPageViewModel Stats { get; }

    /// <summary>Settings page (pinned to the bottom of the navigation).</summary>
    public SettingsPageViewModel Settings { get; }

    /// <summary>Update checks and the "update available" dialog (the same instance the Settings page shows).</summary>
    public UpdatesViewModel Updates => Settings.Updates;

    /// <summary>The dialog that asks to accept licence terms before a download.</summary>
    public LicenseDialogViewModel LicenseDialog { get; }

    /// <summary>The About dialog.</summary>
    public AboutViewModel About { get; }

    /// <summary>The one-time offer to move the models of a previous store.</summary>
    public StoreMigrationViewModel StoreMigration { get; }

    /// <summary>
    /// Why the storage location asked for is not in use (a portable folder that is not writable, a custom folder that is missing), shown
    /// until dismissed; null otherwise.
    /// </summary>
    [ObservableProperty]
    public partial string? StorageNotice { get; set; }

    /// <summary>The download queue, for the indicator shown on every page.</summary>
    public DownloadsViewModel Downloads => Models.Downloads;

    /// <summary>The grouped navigation and the routing from a model kind to its page.</summary>
    public NavigationViewModel Navigation { get; }

    /// <summary>The pages of the navigation in its order, each once (the Settings page excepted: the application menu has its own item).</summary>
    public ObservableCollection<PageViewModel> Pages { get; }

    /// <summary>The visible page.</summary>
    [ObservableProperty]
    public partial PageViewModel? CurrentPage { get; set; }

    /// <summary>Loaded models, for the status bar.</summary>
    [ObservableProperty]
    public partial string LoadedText { get; set; } = "";

    /// <summary>Server state, for the status bar.</summary>
    [ObservableProperty]
    public partial string ServerText { get; set; } = "";

    /// <summary>True while the server listens.</summary>
    [ObservableProperty]
    public partial bool IsServing { get; set; }

    /// <summary>The main device, for the status bar.</summary>
    public string DeviceText => _services.Runtime.Devices.FirstOrDefault()?.Name ?? "No device";

    /// <summary>Offers to move the models of a previous store, then shows the first page: the welcome flow on a first start, else the chat or the models.</summary>
    public async Task StartAsync()
    {
        StoreMigration.Check();
        StorageNotice = _services.StorageLocation.Current.Notice;
        _ = Updates.StartAsync();
        if (_services.Runtime.LoadedModels.Count == 0 && !_services.Settings.Current.FirstRunDone)
        {
            var library = await _services.Library.ListAsync(CancellationToken.None);
            if (library.Count == 0)
            {
                await NavigateAsync(Welcome);
                return;
            }

            _services.Settings.Update(settings => settings with { FirstRunDone = true });
        }

        await NavigateAsync(_services.Runtime.LoadedModels.Count == 0 ? Models : Chat);
    }

    [RelayCommand]
    private void DismissStorageNotice() => StorageNotice = null;

    /// <summary>Shows the About dialog (application menu on macOS).</summary>
    [RelayCommand]
    public void ShowAbout() => About.IsOpen = true;

    /// <summary>Shows the Settings page.</summary>
    [RelayCommand]
    public Task OpenSettingsAsync() => NavigateAsync(Settings);

    /// <summary>Shows the Settings page and checks for an update now.</summary>
    [RelayCommand]
    public async Task CheckForUpdatesAsync()
    {
        Settings.ShowTab(SettingsTab.Updates);
        await NavigateAsync(Settings);
        await Updates.CheckAsync(manual: true);
    }

    /// <summary>Opens a link (documentation, issue tracker) in the browser.</summary>
    [RelayCommand]
    public void OpenLink(string? url)
    {
        if (!string.IsNullOrEmpty(url))
        {
            _services.Platform.Open(url);
        }
    }

    [RelayCommand]
    private Task OpenStorageSettingsAsync()
    {
        StorageNotice = null;
        Settings.ShowTab(SettingsTab.Storage);
        return NavigateAsync(Settings);
    }

    /// <inheritdoc/>
    public void OpenChat(ILoadedModel? model)
    {
        Chat.SelectModel(model);
        _ = NavigateAsync(Chat);
    }

    /// <inheritdoc/>
    public void OpenChat(ILoadedModel? model, string? suggestedPrompt) => _ = OpenNewChatAsync(model, suggestedPrompt);

    /// <inheritdoc/>
    public void OpenWelcome()
    {
        _ = NavigateAsync(Welcome);
    }

    private async Task OpenNewChatAsync(ILoadedModel? model, string? suggestedPrompt)
    {
        await NavigateAsync(Chat);
        await Chat.StartChatAsync(model, suggestedPrompt);
    }

    /// <inheritdoc/>
    public bool OpenKind(string kind, ILoadedModel? model)
    {
        var item = Navigation.ForKind(kind);
        switch (item?.Page)
        {
            case ChatPageViewModel:
                OpenChat(model);
                return true;
            case InferencePageViewModel page:
                page.SelectModel(model);
                _ = NavigateAsync(page);
                return true;
            default:
                return false;
        }
    }

    /// <inheritdoc/>
    public void OpenLibrary()
    {
        Models.SelectedTab = ModelsPageViewModel.LibraryTab;
        _ = NavigateAsync(Models);
    }

    /// <inheritdoc/>
    public void OpenLibrary(string kind)
    {
        Models.Library.SelectKind(kind);
        OpenLibrary();
    }

    /// <inheritdoc/>
    public void OpenDiscover(Sharplight.Hub.ModelTask task)
    {
        Models.Discover.Filters.Apply(Models.Discover.Filters.Current with { Task = task });
        OpenDiscover();
    }

    /// <inheritdoc/>
    public void OpenVerifiedModels(string kind)
    {
        Models.Discover.ShowVerifiedKind(kind);
        if (Models.Discover.Filters.Current.Task is not null)
        {
            // The kind has no Hub task: a task left from an earlier visit would select another kind's list.
            Models.Discover.Filters.Apply(Models.Discover.Filters.Current with { Task = null });
        }

        OpenDiscover();
    }

    /// <summary>Shows the resident models.</summary>
    [RelayCommand]
    public void OpenResident()
    {
        Models.SelectedTab = ModelsPageViewModel.ResidentTab;
        _ = NavigateAsync(Models);
    }

    /// <summary>Shows the downloads.</summary>
    [RelayCommand]
    public void OpenDownloads()
    {
        Models.SelectedTab = ModelsPageViewModel.DownloadsTab;
        _ = NavigateAsync(Models);
    }

    /// <inheritdoc/>
    public void OpenDiscover()
    {
        Models.SelectedTab = ModelsPageViewModel.DiscoverTab;
        _ = NavigateAsync(Models);
    }

    /// <summary>Shows <paramref name="page"/>.</summary>
    [RelayCommand]
    public async Task NavigateAsync(PageViewModel page)
    {
        if (ReferenceEquals(page, CurrentPage))
        {
            return;
        }

        CurrentPage?.Deactivate();
        CurrentPage = page;
        await page.ActivateAsync();
    }

    private void UpdateStatus()
    {
        var models = _services.Runtime.LoadedModels;
        int loading = models.Count(model => model.State == Sharplight.Studio.Models.ResidentState.Loading);
        LoadedText = models.Count switch
        {
            0 => "No model loaded",
            1 => loading == 1 ? $"Loading {models[0].Id}" : models[0].Id,
            _ => loading > 0 ? $"{models.Count} models, {loading} loading" : $"{models.Count} models resident",
        };
        IsServing = _services.Server.IsRunning;
        ServerText = IsServing ? _services.Server.BaseUrl ?? "Serving" : "Server off";
        Server.Badge = 0;
    }
}
