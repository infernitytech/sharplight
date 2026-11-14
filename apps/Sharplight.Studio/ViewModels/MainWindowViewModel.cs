using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Studio.Services;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.ViewModels.Chat;
using Sharplight.Studio.ViewModels.Models;
using Sharplight.Studio.ViewModels.Server;
using Sharplight.Studio.ViewModels.Settings;
using Sharplight.Studio.ViewModels.Stats;

namespace Sharplight.Studio.ViewModels;

/// <summary>The main window: navigation between the pages and the status bar.</summary>
internal sealed partial class MainWindowViewModel : ObservableObject, INavigator
{
    private readonly StudioServices _services;

    public MainWindowViewModel(StudioServices services)
    {
        _services = services;
        Chat = new ChatPageViewModel(services, this);
        Models = new ModelsPageViewModel(services, this);
        Server = new ServerPageViewModel(services);
        Stats = new StatsPageViewModel(services);
        Settings = new SettingsPageViewModel(services);
        Pages = [Chat, Models, Server, Stats];
        services.Runtime.ModelsChanged += (_, _) => services.Dispatcher.Post(UpdateStatus);
        services.Server.StateChanged += (_, _) => services.Dispatcher.Post(UpdateStatus);
        UpdateStatus();
    }

    /// <summary>Chat page.</summary>
    public ChatPageViewModel Chat { get; }

    /// <summary>Models page.</summary>
    public ModelsPageViewModel Models { get; }

    /// <summary>Server page.</summary>
    public ServerPageViewModel Server { get; }

    /// <summary>Statistics page.</summary>
    public StatsPageViewModel Stats { get; }

    /// <summary>Settings page (pinned to the bottom of the navigation).</summary>
    public SettingsPageViewModel Settings { get; }

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

    /// <summary>Shows the first page.</summary>
    public Task StartAsync() => NavigateAsync(_services.Runtime.LoadedModels.Count == 0 ? Models : Chat);

    /// <inheritdoc/>
    public void OpenChat(ILoadedModel? model)
    {
        Chat.SelectModel(model);
        _ = NavigateAsync(Chat);
    }

    /// <inheritdoc/>
    public void OpenLibrary()
    {
        Models.SelectedTab = ModelsPageViewModel.LibraryTab;
        _ = NavigateAsync(Models);
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
