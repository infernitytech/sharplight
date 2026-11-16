using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Studio.Services;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.ViewModels.Chat;
using Sharplight.Studio.ViewModels.Image;
using Sharplight.Studio.ViewModels.Inference;
using Sharplight.Studio.ViewModels.Models;
using Sharplight.Studio.ViewModels.Server;
using Sharplight.Studio.ViewModels.Settings;
using Sharplight.Studio.ViewModels.Audio;
using Sharplight.Studio.ViewModels.Speech;
using Sharplight.Studio.ViewModels.Stats;
using Sharplight.Studio.ViewModels.Tasks;
using Sharplight.Studio.ViewModels.Updates;
using Sharplight.Studio.ViewModels.Vision;

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
        Audio = new AudioPageViewModel(services, this);
        Image = new ImagePageViewModel(services, this);
        Vision = new VisionPageViewModel(services, this);
        Tasks = new TasksPageViewModel(services, this);
        Models = new ModelsPageViewModel(services, this);
        Server = new ServerPageViewModel(services);
        Stats = new StatsPageViewModel(services);
        Settings = new SettingsPageViewModel(services);
        StoreMigration = new StoreMigrationViewModel(services, () => _ = Models.Library.RefreshAsync());
        Pages = [Chat, Transcribe, Speak, Audio, Image, Vision, Tasks, Models, Server, Stats];
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

    /// <summary>Audio page: audio codecs and voice activity detection.</summary>
    public AudioPageViewModel Audio { get; }

    /// <summary>Image page: image generation.</summary>
    public ImagePageViewModel Image { get; }

    /// <summary>Vision page: image classification, embedding, object detection and depth estimation.</summary>
    public VisionPageViewModel Vision { get; }

    /// <summary>Tasks page: the model kinds and where each stands.</summary>
    public TasksPageViewModel Tasks { get; }

    /// <summary>Models page.</summary>
    public ModelsPageViewModel Models { get; }

    /// <summary>Server page.</summary>
    public ServerPageViewModel Server { get; }

    /// <summary>Statistics page.</summary>
    public StatsPageViewModel Stats { get; }

    /// <summary>Settings page (pinned to the bottom of the navigation).</summary>
    public SettingsPageViewModel Settings { get; }

    /// <summary>Update checks and the "update available" dialog (the same instance the Settings page shows).</summary>
    public UpdatesViewModel Updates => Settings.Updates;

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

    /// <summary>Main navigation entries.</summary>
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

    /// <summary>Offers to move the models of a previous store, then shows the first page.</summary>
    public Task StartAsync()
    {
        StoreMigration.Check();
        StorageNotice = _services.StorageLocation.Current.Notice;
        _ = Updates.StartAsync();
        return NavigateAsync(_services.Runtime.LoadedModels.Count == 0 ? Models : Chat);
    }

    [RelayCommand]
    private void DismissStorageNotice() => StorageNotice = null;

    [RelayCommand]
    private Task OpenStorageSettingsAsync()
    {
        StorageNotice = null;
        return NavigateAsync(Settings);
    }

    /// <inheritdoc/>
    public void OpenChat(ILoadedModel? model)
    {
        Chat.SelectModel(model);
        _ = NavigateAsync(Chat);
    }

    /// <inheritdoc/>
    public bool OpenKind(string kind, ILoadedModel? model)
    {
        InferencePageViewModel? page = kind switch
        {
            Sharplight.Models.ModelKinds.SpeechRecognition => Transcribe,
            Sharplight.Models.ModelKinds.SpeechSynthesis => Speak,
            Sharplight.Models.ModelKinds.AudioCodec or Sharplight.Models.ModelKinds.VoiceActivity => Audio,
            Sharplight.Models.ModelKinds.ImageGeneration => Image,
            Sharplight.Models.ModelKinds.ImageClassification or Sharplight.Models.ModelKinds.ImageEmbedding
                or Sharplight.Models.ModelKinds.ObjectDetection or Sharplight.Models.ModelKinds.DepthEstimation => Vision,
            _ => null,
        };
        if (page is null)
        {
            if (kind == Sharplight.Models.ModelKinds.TextGeneration)
            {
                OpenChat(model);
                return true;
            }

            return false;
        }

        page.SelectModel(model);
        _ = NavigateAsync(page);
        return true;
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
