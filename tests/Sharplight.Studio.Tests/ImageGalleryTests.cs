using Avalonia.Threading;
using Sharplight.ImageGeneration;
using Sharplight.Models;
using Sharplight.Studio.Services.Gallery;
using Sharplight.Studio.Services.Imaging;
using Sharplight.Studio.Tests.Fakes;
using Sharplight.Studio.ViewModels.Gallery;
using Sharplight.Studio.ViewModels.Image;
using Sharplight.Studio.ViewModels.Inference.Parameters;

namespace Sharplight.Studio.Tests;

/// <summary>The image gallery: storage in the data folder, the browsing view model and its use on the Image page.</summary>
public sealed class ImageGalleryTests
{
    private static byte[] Png(int tint = 1) => ImageFiles.ToPng(FakeImageGenerator.Gradient(8, 8, tint));

    private static GalleryEntry Entry(string prompt, string model = "sd-tiny", long seed = 1, DateTimeOffset? created = null) => new()
    {
        Id = ImageGalleryStore.NewId(),
        Created = created ?? DateTimeOffset.Now,
        Model = model,
        Prompt = prompt,
        NegativePrompt = "blurry",
        Seed = seed,
        Steps = 12,
        Guidance = 6,
        Width = 8,
        Height = 8,
        Sampler = "euler",
        Seconds = 1.5,
        Options = new Dictionary<string, string> { ["steps"] = "12", ["guidance"] = "6", ["sampler"] = "euler" },
    };

    private static ImageGalleryViewModel Gallery(TestServices services, ImageGalleryStore store, List<GalleryItemViewModel>? opened = null, List<GalleryEntry>? reused = null)
        => new(store, services.Platform, services.Platform, item => opened?.Add(item), entry => reused?.Add(entry));

    [Fact]
    public void Store_RoundTripsEntriesNewestFirst_AndDeletes()
    {
        using var services = new TestServices();
        var store = new ImageGalleryStore(services.Services.Pages.GenerationsDirectory("images"));
        var older = store.Add(Entry("old", created: DateTimeOffset.Now.AddDays(-2)), Png(1));
        var newer = store.Add(Entry("new", created: DateTimeOffset.Now), Png(2));

        Assert.Matches(@"^\d{4}-\d{2}-\d{2}/.+\.png$", newer.File);
        Assert.True(File.Exists(store.PathOf(newer)));
        Assert.True(File.Exists(Path.ChangeExtension(store.PathOf(newer), ".json")));
        Assert.StartsWith(services.Directory, store.Folder, StringComparison.Ordinal);

        var loaded = new ImageGalleryStore(store.Folder).Load();
        Assert.Equal(["new", "old"], loaded.Select(entry => entry.Prompt));
        Assert.Equal(newer.Id, loaded[0].Id);
        Assert.Equal("euler", loaded[0].Sampler);
        Assert.Equal("12", loaded[0].Options["steps"]);
        Assert.Equal(Png(2), store.ReadPng(loaded[0]));

        store.Delete(older);
        Assert.Equal(["new"], store.Load().Select(entry => entry.Prompt));
        Assert.False(Directory.Exists(Path.GetDirectoryName(store.PathOf(older))));
    }

    [Fact]
    public void Store_SkipsDamagedSidecars_AndOrphans()
    {
        using var services = new TestServices();
        var store = new ImageGalleryStore(services.Services.Pages.GenerationsDirectory("images"));
        var good = store.Add(Entry("good"), Png());
        File.WriteAllText(Path.ChangeExtension(store.PathOf(good), ".json").Replace(good.Id, "broken"), "{ nope");
        File.WriteAllBytes(store.PathOf(good).Replace(good.Id, "broken"), Png());
        File.WriteAllText(store.PathOf(good).Replace(good.Id, "orphan").Replace(".png", ".json"), "{}");
        Assert.Equal(["good"], store.Load().Select(entry => entry.Prompt));
    }

    [Fact]
    public async Task ViewModel_FiltersByModelAndText_AndOpensAndReuses()
    {
        using var services = new TestServices();
        var store = new ImageGalleryStore(services.Services.Pages.GenerationsDirectory("images"));
        store.Add(Entry("a lighthouse at dusk", "sd-tiny", 1, DateTimeOffset.Now.AddMinutes(-3)), Png());
        store.Add(Entry("a red bicycle", "flux", 2, DateTimeOffset.Now.AddMinutes(-2)), Png());
        store.Add(Entry("a Lighthouse in fog", "flux", 3, DateTimeOffset.Now.AddMinutes(-1)), Png());
        var opened = new List<GalleryItemViewModel>();
        var reused = new List<GalleryEntry>();
        var gallery = Gallery(services, store, opened, reused);
        await gallery.LoadAsync();

        Assert.Equal(3, gallery.Count);
        Assert.Equal(["a Lighthouse in fog", "a red bicycle", "a lighthouse at dusk"], gallery.Visible.Select(item => item.Entry.Prompt));
        Assert.Equal([ImageGalleryViewModel.AllModels, "flux", "sd-tiny"], gallery.Models);

        gallery.TextFilter = "lighthouse";
        Assert.Equal(2, gallery.Visible.Count);
        Assert.Equal("2 of 3 images", gallery.CountText);
        gallery.ModelFilter = "flux";
        Assert.Equal(["a Lighthouse in fog"], gallery.Visible.Select(item => item.Entry.Prompt));
        gallery.ClearFiltersCommand.Execute(null);
        Assert.Equal(3, gallery.Visible.Count);

        gallery.SelectedItem = gallery.Visible[1];
        Assert.Same(gallery.SelectedItem, Assert.Single(opened));
        gallery.ReuseSettingsCommand.Execute(null);
        Assert.Equal(2, Assert.Single(reused).Seed);
        await gallery.CopyPromptCommand.ExecuteAsync(null);
        Assert.Equal(["a red bicycle"], services.Platform.Copied);
        gallery.OpenFolderCommand.Execute(null);
        Assert.Equal(store.DirectoryOf(gallery.SelectedItem!.Entry), Assert.Single(services.Platform.Opened));
    }

    [Fact]
    public async Task Delete_AsksForConfirmation_ThenRemovesFiles()
    {
        using var services = new TestServices();
        var store = new ImageGalleryStore(services.Services.Pages.GenerationsDirectory("images"));
        var stored = store.Add(Entry("doomed"), Png());
        store.Add(Entry("kept"), Png());
        var gallery = Gallery(services, store);
        await gallery.LoadAsync();
        var item = gallery.Visible.Single(candidate => candidate.Entry.Prompt == "doomed");

        gallery.DeleteCommand.Execute(item);
        Assert.True(item.IsConfirmingDelete);
        Assert.Equal(2, gallery.Count);
        gallery.CancelDeleteCommand.Execute(item);
        Assert.False(item.IsConfirmingDelete);
        gallery.DeleteCommand.Execute(item);
        gallery.DeleteCommand.Execute(item);
        await gallery.Deleting;

        Assert.Equal(["kept"], gallery.Visible.Select(entry => entry.Entry.Prompt));
        Assert.False(File.Exists(store.PathOf(stored)));
        Assert.Equal(["kept"], store.Load().Select(entry => entry.Prompt));
    }

    [Fact]
    public async Task ImagePage_StoresEveryImage_AndSurvivesARestart_AndReusesSettings()
    {
        using var services = new TestServices();
        var client = new FakeImageGenerator();
        var model = new FakeModel("sd-tiny") { Kind = ModelKinds.ImageGeneration };
        model.Clients[typeof(IImageGenerator)] = client;
        services.Runtime.Add(model);
        var page = new ImagePageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        await page.Generations.Loading;
        Assert.True(page.Generations.IsEmpty);

        page.Prompt = "a lighthouse at dusk";
        page.NegativePrompt = "blurry";
        page.Panel.Parameters.OfType<NumberParameterViewModel>().Single(parameter => parameter.Key == "steps").Value = 6;
        page.Panel.Parameters.OfType<NumberParameterViewModel>().Single(parameter => parameter.Key == "count").Value = 2;
        page.Panel.Parameters.OfType<TextParameterViewModel>().Single(parameter => parameter.Key == "seed").Value = "7";
        page.RunCommand.Execute(null);
        await page.Queue.Items.Single().Completion.WaitAsync(TestContext.Current.CancellationToken);

        Assert.Equal(2, page.Generations.Count);
        Assert.Equal([8L, 7L], page.Generations.Visible.Select(item => item.Entry.Seed));
        var stored = page.Generations.Visible[1].Entry;
        Assert.Equal(("sd-tiny", "a lighthouse at dusk", "blurry", 6), (stored.Model, stored.Prompt, stored.NegativePrompt, stored.Steps));
        Assert.Contains(Path.Combine("generations", "images"), new ImageGalleryStore(services.Services.Pages.GenerationsDirectory("images")).PathOf(stored), StringComparison.Ordinal);

        // A new page (a restart) lists the same images from disk.
        var reopened = new ImagePageViewModel(services.Services, new FakeNavigator());
        await reopened.ActivateAsync();
        await reopened.Generations.Loading;
        Assert.Equal([8L, 7L], reopened.Generations.Visible.Select(item => item.Entry.Seed));

        // Reuse settings fills the form, the seed included.
        reopened.Prompt = "";
        reopened.Generations.ReuseSettingsCommand.Execute(reopened.Generations.Visible[1]);
        Assert.Equal("a lighthouse at dusk", reopened.Prompt);
        Assert.Equal("blurry", reopened.NegativePrompt);
        Assert.Equal("7", reopened.Panel.Parameters.OfType<TextParameterViewModel>().Single(parameter => parameter.Key == "seed").Value);
        Assert.Equal(6, reopened.Panel.Parameters.OfType<NumberParameterViewModel>().Single(parameter => parameter.Key == "steps").Value);

        // Opening shows the stored image large (read and decoded on a worker).
        reopened.Generations.SelectedItem = reopened.Generations.Visible[0];
        for (int i = 0; i < 100 && reopened.SelectedImage?.Seed != 8; i++)
        {
            await Task.Delay(20, TestContext.Current.CancellationToken);
        }

        Assert.Equal(8, reopened.SelectedImage!.Seed);
    }

    [Fact]
    public Task OpeningAGalleryOf2000Images_DoesNotBlockTheUiThread() => HeadlessApp.RunAsync(async () =>
    {
        using var services = new TestServices(dispatcher: new Sharplight.Studio.Platform.AvaloniaDispatcher());
        var store = new ImageGalleryStore(services.Services.Pages.GenerationsDirectory("images"));
        byte[] png = Png();
        Parallel.For(0, 2000, new ParallelOptions { MaxDegreeOfParallelism = 4 }, i
            => store.Add(Entry($"prompt number {i}", i % 2 == 0 ? "sd-tiny" : "flux", i, DateTimeOffset.Now.AddSeconds(-i)), png));

        var gallery = Gallery(services, store);
        var heartbeat = UiHeartbeat.Start();
        var loading = gallery.LoadAsync();
        await loading.WaitAsync(TimeSpan.FromSeconds(60), TestContext.Current.CancellationToken);
        for (int i = 0; i < 40; i++)
        {
            await Task.Delay(25, TestContext.Current.CancellationToken);
            Dispatcher.UIThread.RunJobs();
        }

        gallery.TextFilter = "number 19";
        gallery.ModelFilter = "flux";
        heartbeat.StopAndAssertResponsive("gallery of 2000 images");
        Assert.Equal(2000, gallery.Count);
        Assert.NotEmpty(gallery.Visible);
        Assert.All(gallery.Visible, item => Assert.Equal("flux", item.Entry.Model));
    });
}
