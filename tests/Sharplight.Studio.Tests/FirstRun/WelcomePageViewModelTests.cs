using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Library;
using Sharplight.Studio.Tests.Fakes;
using Sharplight.Studio.ViewModels;
using Sharplight.Studio.ViewModels.Models;
using Sharplight.Studio.ViewModels.Welcome;

namespace Sharplight.Studio.Tests.FirstRun;

/// <summary>The welcome flow over a local test feed: detection, recommendation, install with progress, load, new chat, skip, offline and no GPU.</summary>
public sealed class WelcomePageViewModelTests
{
    private sealed class Flow : IDisposable
    {
        public Flow(AppSettings? settings = null, long? freeDisk = 200L << 30)
        {
            Rig = new FeedCatalogTests.Rig(settings);
            Navigator = new FakeNavigator();
            Models = new ModelsPageViewModel(Rig.All, Navigator);
            Welcome = new WelcomePageViewModel(Rig.All, Navigator, Models, _ => freeDisk);
            Models.Discover.Feed.Installed += (_, _) => ListInstalledChat();
        }

        public FeedCatalogTests.Rig Rig { get; }

        public FakeNavigator Navigator { get; }

        public ModelsPageViewModel Models { get; }

        public WelcomePageViewModel Welcome { get; }

        public async Task OpenAsync()
        {
            await Welcome.ActivateAsync();
            await Welcome.Loading;
        }

        /// <summary>Makes the library list the chat package once it is in the store, as the real library does.</summary>
        private void ListInstalledChat()
        {
            var stored = Rig.Store.List().FirstOrDefault(model => model.FeedPackage == WelcomeFeed.Chat);
            if (stored is null || Rig.Services.Library.Entries.Count > 0)
            {
                return;
            }

            Rig.Services.Library.Entries.Add(new LibraryEntry
            {
                Id = "starter-chat",
                DisplayName = "Starter Chat",
                Path = "/models/starter-chat/model.slm",
                Format = "slm",
                Manifest = stored,
            });
        }

        public void Dispose() => Rig.Dispose();
    }

    [Fact]
    public async Task Open_DetectsTheMachine_AndRecommendsChatAndSpeech()
    {
        using var flow = new Flow();
        await WelcomeFeed.PublishAsync(flow.Rig);

        await flow.OpenAsync();

        var welcome = flow.Welcome;
        Assert.Equal("Test GPU · 16.0 GB", welcome.GpuText);
        Assert.Equal("32.0 GB", welcome.MemoryText);
        Assert.Contains("200 GB free", welcome.DiskText, StringComparison.Ordinal);
        Assert.Equal([WelcomeFeed.Chat, WelcomeFeed.Speech, WelcomeFeed.Embedding], welcome.Items.Select(item => item.Package.Id));
        Assert.Equal([true, true, false], welcome.Items.Select(item => item.IsSelected));
        Assert.True(welcome.Items[0].IsRequired);
        Assert.Contains("Runs on your GPU", welcome.Items[0].FitText, StringComparison.Ordinal);
        Assert.StartsWith("Download ", welcome.TotalText, StringComparison.Ordinal);
        Assert.Contains("under a minute", welcome.TotalText, StringComparison.Ordinal);
        Assert.True(welcome.CanStart);
        Assert.False(welcome.HasMessage);

        welcome.Items[2].IsSelected = true;
        Assert.DoesNotContain("Nothing", welcome.TotalText, StringComparison.Ordinal);
    }

    [Fact]
    public async Task GetStarted_InstallsLoadsTheChatModel_AndOpensANewChatWithAnExample()
    {
        using var flow = new Flow();
        await WelcomeFeed.PublishAsync(flow.Rig);
        await flow.OpenAsync();
        var progress = new List<double>();
        flow.Welcome.PropertyChanged += (_, args) =>
        {
            if (args.PropertyName == nameof(WelcomePageViewModel.Progress))
            {
                progress.Add(flow.Welcome.Progress);
            }
        };

        await flow.Welcome.GetStartedAsync();

        Assert.Null(flow.Welcome.Error);
        var stored = flow.Rig.Store.List().Select(model => model.FeedPackage).Order().ToList();
        Assert.Equal([WelcomeFeed.Chat, WelcomeFeed.Speech, WelcomeFeed.Detector], stored);
        var request = Assert.Single(flow.Rig.Services.Runtime.Requests);
        Assert.Equal("/models/starter-chat/model.slm", request.Path);
        Assert.Equal(Sharplight.Models.ModelKinds.TextGeneration, request.Kind);
        Assert.Equal(FakeRuntime.Gpu.Id, request.Settings.DeviceId);
        Assert.Equal(1, request.Settings.Slots);
        Assert.True(flow.Rig.Services.Settings.Current.FirstRunDone);
        Assert.Equal(["chat", "chat:new"], flow.Navigator.Visited);
        Assert.Equal(request.Id, flow.Navigator.LastChatModel?.Id);
        Assert.False(string.IsNullOrWhiteSpace(Assert.Single(flow.Navigator.SuggestedPrompts)));
        Assert.Equal(1, flow.Welcome.Progress);
        Assert.Equal(progress.Order(), progress);
        Assert.All(flow.Welcome.Items.Where(item => item.IsSelected), item => Assert.Equal("Ready", item.State));
        Assert.False(flow.Welcome.IsWorking);
    }

    [Fact]
    public async Task GetStarted_WithOnlyChatSelected_InstallsOnlyTheChatModel()
    {
        using var flow = new Flow();
        await WelcomeFeed.PublishAsync(flow.Rig);
        await flow.OpenAsync();
        flow.Welcome.Items[1].IsSelected = false;

        await flow.Welcome.GetStartedAsync();

        Assert.Equal([WelcomeFeed.Chat], flow.Rig.Store.List().Select(model => model.FeedPackage));
    }

    [Fact]
    public async Task Skip_GoesToDiscover_AndIsRemembered()
    {
        using var flow = new Flow();
        await WelcomeFeed.PublishAsync(flow.Rig);
        await flow.OpenAsync();

        flow.Welcome.SkipCommand.Execute(null);

        Assert.Equal(["discover"], flow.Navigator.Visited);
        Assert.True(flow.Rig.Services.Settings.Current.FirstRunDone);
        Assert.Empty(flow.Rig.Store.List());
    }

    [Fact]
    public async Task Offline_WithoutASavedListing_ExplainsAndOffersTryAgain()
    {
        using var flow = new Flow();
        await WelcomeFeed.PublishAsync(flow.Rig);
        flow.Rig.Offline = true;

        await flow.OpenAsync();

        Assert.False(flow.Welcome.HasItems);
        Assert.True(flow.Welcome.ShowEmpty);
        Assert.Contains("internet", flow.Welcome.Message, StringComparison.Ordinal);
        Assert.False(flow.Welcome.CanStart);

        flow.Rig.Offline = false;
        await flow.Welcome.RetryAsync();

        Assert.True(flow.Welcome.HasItems);
        Assert.False(flow.Welcome.HasMessage);
    }

    [Fact]
    public async Task Offline_WithASavedListing_StillRecommendsAndSaysSo()
    {
        using var flow = new Flow();
        await WelcomeFeed.PublishAsync(flow.Rig);
        await flow.Rig.Service.GetListingAsync(TestContext.Current.CancellationToken);
        flow.Rig.Offline = true;

        await flow.OpenAsync();

        Assert.True(flow.Welcome.IsOffline);
        Assert.True(flow.Welcome.HasItems);
        Assert.Contains("last saved list", flow.Welcome.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Install_WhenTheNetworkDrops_FailsClearly_AndTryAgainResumes()
    {
        using var flow = new Flow();
        await WelcomeFeed.PublishAsync(flow.Rig);
        await flow.OpenAsync();
        flow.Rig.Offline = true;

        await flow.Welcome.GetStartedAsync();

        Assert.True(flow.Welcome.HasError);
        Assert.Equal("Try again", flow.Welcome.StartText);
        Assert.False(flow.Welcome.IsWorking);
        Assert.False(flow.Rig.Services.Settings.Current.FirstRunDone);
        Assert.Empty(flow.Navigator.Visited);

        flow.Rig.Offline = false;
        await flow.Welcome.GetStartedAsync();

        Assert.Null(flow.Welcome.Error);
        Assert.Equal("Get started", flow.Welcome.StartText);
        Assert.Equal(["chat", "chat:new"], flow.Navigator.Visited);
    }

    [Fact]
    public async Task WithoutAGpu_TheRecommendationIsCpuSized()
    {
        using var flow = new Flow();
        await WelcomeFeed.PublishAsync(flow.Rig, chatMemory: 5L << 30);
        flow.Rig.Services.Runtime.Devices = [FakeRuntime.Cpu];
        await flow.OpenAsync();

        Assert.Equal("No GPU found: models run on the CPU", flow.Welcome.GpuText);
        Assert.False(flow.Welcome.HasItems);
        Assert.True(flow.Welcome.ShowEmpty);
        Assert.Contains("memory", flow.Welcome.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task WithoutAGpu_AModelThatFitsTheCpuBudgetIsRecommended()
    {
        using var flow = new Flow();
        await WelcomeFeed.PublishAsync(flow.Rig, chatMemory: 1L << 30);
        flow.Rig.Services.Runtime.Devices = [FakeRuntime.Cpu];
        await flow.OpenAsync();

        Assert.True(flow.Welcome.IsCpuOnly);
        Assert.Contains("Runs on the CPU", flow.Welcome.Items[0].FitText, StringComparison.Ordinal);
    }

    [Fact]
    public async Task FirstStart_ShowsTheWelcomeFlow_OnlyWhileTheLibraryIsEmptyAndTheFlowIsNotDone()
    {
        using var rig = new FeedCatalogTests.Rig();
        var window = new MainWindowViewModel(rig.All);

        await window.StartAsync();

        Assert.Same(window.Welcome, window.CurrentPage);

        rig.Services.Settings.Update(settings => settings with { FirstRunDone = true });
        var again = new MainWindowViewModel(rig.All);
        await again.StartAsync();

        Assert.Same(again.Models, again.CurrentPage);
    }

    [Fact]
    public async Task FirstStart_WithModelsInTheLibrary_SkipsTheWelcomeFlowAndRemembersIt()
    {
        using var rig = new FeedCatalogTests.Rig();
        rig.Services.Library.Entries.Add(new LibraryEntry { Id = "m", DisplayName = "m", Path = "/models/m.gguf", Format = "gguf" });
        var window = new MainWindowViewModel(rig.All);

        await window.StartAsync();

        Assert.NotSame(window.Welcome, window.CurrentPage);
        Assert.True(rig.Services.Settings.Current.FirstRunDone);
    }

    [Fact]
    public async Task ChatPage_WithoutAModel_OffersGetAModel()
    {
        using var rig = new FeedCatalogTests.Rig();
        var navigator = new FakeNavigator();
        var chat = new Sharplight.Studio.ViewModels.Chat.ChatPageViewModel(rig.All, navigator);
        await chat.ActivateAsync();

        Assert.True(chat.ShowGetModel);
        chat.GetModelCommand.Execute(null);

        Assert.Equal(["welcome"], navigator.Visited);

        rig.Services.Runtime.Add(new FakeModel("m"));
        chat.Picker.Select(null);
        await HeadlessWait(() => chat.HasModels);
        Assert.False(chat.ShowGetModel);
    }

    private static async Task HeadlessWait(Func<bool> condition)
    {
        for (int attempt = 0; attempt < 100 && !condition(); attempt++)
        {
            await Task.Delay(20, TestContext.Current.CancellationToken);
        }
    }
}
