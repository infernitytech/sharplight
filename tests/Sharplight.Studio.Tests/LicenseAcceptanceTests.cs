using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Logging;
using Avalonia.Media.Imaging;
using Avalonia.Threading;
using Sharplight.Hub;
using Sharplight.Hub.Catalog;
using Sharplight.ModelFeed;
using Sharplight.Studio.Models;
using Sharplight.Studio.Platform;
using Sharplight.Studio.Services.Library;
using Sharplight.Studio.ViewModels;
using Sharplight.Studio.ViewModels.Models;
using Sharplight.Studio.Views;

namespace Sharplight.Studio.Tests;

/// <summary>The "accept the licence terms" step before a download: which models need it, that every download path asks, that the answer is remembered, and the dialog.</summary>
public sealed class LicenseAcceptanceTests
{
    private const string Commit = "0123456789abcdef0123456789abcdef01234567";
    private const string Sha = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    private static CatalogEntry Restricted(string license = "other", string? notice = "Non-commercial use only.") => new()
    {
        Id = "omni", Name = "Omni", Kind = "text-generation", License = license, LicenseNotice = notice,
        Variants =
        [
            new CatalogVariant
            {
                Id = "q8-0", Label = "Q8_0", Format = "gguf", Quantization = "Q8_0", Repository = "owner/Omni-GGUF", Revision = Commit, Primary = "model.gguf",
                Files = [new CatalogFile { Path = "model.gguf", Size = 600, Sha256 = Sha }], MemoryBytes = 1L << 30, Recommended = true,
            },
        ],
        Verification = new CatalogVerification { Summary = "identical", Providers = ["CPU"], Date = "2026-11-16" },
    };

    private static LicenseTerms Terms(string key = "omni", string version = "v1") => new(key, "Omni", "other", "Non-commercial use only.", "https://example.test/license", version);

    [Theory]
    [InlineData("apache-2.0", false)]
    [InlineData("MIT", false)]
    [InlineData("bsd-3-clause", false)]
    [InlineData("bsd-2-clause-patent", false)]
    [InlineData("cc-by-4.0", false)]
    [InlineData("cc-by-nc-4.0", true)]
    [InlineData("other", true)]
    [InlineData("llama3.3", true)]
    [InlineData(null, false)]
    public void TermsFor_AHubRepository_AreRequiredForNonPermissiveLicences_AndGatedRepositories(string? license, bool required)
    {
        Assert.Equal(required, LicenseNotices.TermsFor("owner/repo", license, gated: false) is not null);
        Assert.NotNull(LicenseNotices.TermsFor("owner/repo", license, gated: true));
    }

    [Fact]
    public void TermsFor_ACatalogEntry_UsesTheNoticeAndLinksTheRepository()
    {
        var terms = LicenseNotices.TermsFor(Restricted())!;

        Assert.Equal(("omni", "Omni", "other", "Non-commercial use only."), (terms.Key, terms.ModelName, terms.License, terms.Notice));
        Assert.Equal("https://huggingface.co/owner/Omni-GGUF", terms.Url);
        Assert.NotEqual(terms.Version, LicenseNotices.TermsFor(Restricted(notice: "Changed."))!.Version);
        Assert.Null(LicenseNotices.TermsFor(Restricted("apache-2.0", null)));
        Assert.NotNull(LicenseNotices.TermsFor(Restricted("apache-2.0", "Attribution required.")));
    }

    [Fact]
    public void TermsFor_AFeedPackage_NeedsTermsForGatedNonCommercialOrRestrictiveLicences()
    {
        Assert.Null(LicenseNotices.TermsFor("p", "P", new PackageLicense { Spdx = "mit" }));
        Assert.Null(LicenseNotices.TermsFor("p", "P", null));
        Assert.NotNull(LicenseNotices.TermsFor("p", "P", new PackageLicense { Spdx = "mit", Gated = true }));
        Assert.NotNull(LicenseNotices.TermsFor("p", "P", new PackageLicense { Spdx = "mit", CommercialUse = false }));
        var terms = LicenseNotices.TermsFor("p", "P", new PackageLicense { Spdx = "other", Name = "Vendor licence", Url = "https://vendor.test/l", Attribution = "Built with Vendor." })!;
        Assert.Equal(("feed:p", "Vendor licence", "Built with Vendor.", "https://vendor.test/l"), (terms.Key, terms.License, terms.Notice, terms.Url));
    }

    [Fact]
    public async Task Gate_AsksOnce_AndRemembersTheAcceptanceInTheSettings()
    {
        using var services = new TestServices();
        var gate = services.Licenses;

        Assert.True(await gate.RequestAsync(null));
        var first = gate.RequestAsync(Terms());
        Assert.False(first.IsCompleted);
        Assert.Equal("omni", gate.Pending!.Key);
        gate.Accept();

        Assert.True(await first);
        Assert.Null(gate.Pending);
        var saved = Assert.Single(services.Settings.Current.AcceptedLicenses);
        Assert.Equal(("omni", "other", "v1"), (saved.Key, saved.License, saved.TermsVersion));
        Assert.True((await gate.RequestAsync(Terms())));
        Assert.True(gate.HasAccepted("omni"));

        // Changed terms are asked again.
        var again = gate.RequestAsync(Terms(version: "v2"));
        Assert.False(again.IsCompleted);
        gate.Decline();
        Assert.False(await again);
        Assert.Equal("v1", Assert.Single(services.Settings.Current.AcceptedLicenses).TermsVersion);
    }

    [Fact]
    public async Task Gate_QueuesRequests_AnswersEqualOnesTogether_AndWithdrawsCancelledOnes()
    {
        using var services = new TestServices();
        var gate = services.Licenses;
        using var cancel = new CancellationTokenSource();
        var a1 = gate.RequestAsync(Terms());
        var a2 = gate.RequestAsync(Terms());
        var b = gate.RequestAsync(Terms("other-model"), cancel.Token);

        gate.Accept();
        Assert.True(await a1 && await a2);
        Assert.Equal("other-model", gate.Pending!.Key);

        cancel.Cancel();
        Assert.False(await b);
        Assert.Null(gate.Pending);
    }

    [Fact]
    public async Task CatalogInstall_WaitsForTheAcceptance_AndDownloadsAfterwards()
    {
        using var services = new TestServices();
        services.Hub.Catalog = new VerifiedCatalog { Updated = "2026-11-16", Entries = [Restricted()] };
        var downloads = new DownloadsViewModel(services.Services);
        var search = new HubSearchViewModel(services.Services, downloads, () => new HashSet<string>());
        await search.Verified.LoadAsync();
        search.Verified.SelectedEntry = search.Verified.Entries.Single();

        var install = search.Verified.InstallCommand.ExecuteAsync(null);
        Assert.False(install.IsCompleted);
        Assert.Empty(downloads.Items);
        Assert.Equal("Omni", services.Licenses.Pending!.ModelName);

        services.Licenses.Accept();
        await install;
        await services.Hub.Started.Task.WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);
        Assert.Equal("omni/q8-0", Assert.Single(downloads.Items).CatalogKey);
        services.Hub.Release();
    }

    [Fact]
    public async Task CatalogInstall_Declined_DownloadsNothing_AndAcceptedModelsAreNotAskedAgain()
    {
        using var services = new TestServices();
        services.Hub.Catalog = new VerifiedCatalog { Updated = "2026-11-16", Entries = [Restricted()] };
        var downloads = new DownloadsViewModel(services.Services);
        var search = new HubSearchViewModel(services.Services, downloads, () => new HashSet<string>());
        await search.Verified.LoadAsync();
        search.Verified.SelectedEntry = search.Verified.Entries.Single();

        var declined = search.Verified.InstallCommand.ExecuteAsync(null);
        services.Licenses.Decline();
        await declined;
        Assert.Empty(downloads.Items);
        Assert.Equal(0, services.Hub.Pulls);

        var accepted = search.Verified.InstallCommand.ExecuteAsync(null);
        services.Licenses.Accept();
        await accepted;
        await services.Hub.Started.Task.WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);
        services.Hub.Release();

        // A second install of the same model asks nothing.
        Assert.True(services.Licenses.IsAccepted(LicenseNotices.TermsFor(Restricted())!));
        services.Hub.Reset();
        await search.Verified.InstallCommand.ExecuteAsync(null);
        Assert.Null(services.Licenses.Pending);
    }

    [Fact]
    public async Task HubDownload_OfARestrictiveRepository_AsksFirst()
    {
        using var services = new TestServices();
        var downloads = new DownloadsViewModel(services.Services);
        var search = new HubSearchViewModel(services.Services, downloads, () => new HashSet<string>());
        services.Hub.Results.Add(new HubModelSummary("owner/Restricted-GGUF", 10, 1, false, "gguf", "text-generation", null, [ModelFormat.Gguf], ["Q8_0"], License: "cc-by-nc-4.0"));
        search.Query = "restricted";
        await search.SearchCommand.ExecuteAsync(null);
        await search.FitEstimate;
        Assert.NotNull(search.SelectedVariant);

        var download = search.DownloadCommand.ExecuteAsync(null);
        Assert.False(download.IsCompleted);
        Assert.Equal("hub:owner/Restricted-GGUF", services.Licenses.Pending!.Key);
        Assert.Contains("cc-by-nc-4.0", services.Licenses.Pending.Notice, StringComparison.Ordinal);
        services.Licenses.Decline();
        await download;

        Assert.Empty(downloads.Items);
    }

    [Fact]
    public async Task FeedInstall_Declined_IsCancelledWithoutInstalling_AndRetryAsksAgain()
    {
        using var rig = new FeedCatalogTests.Rig();
        await FeedCatalogTests.PublishAlphaAsync(rig);
        await rig.Catalog.LoadAsync();
        var answers = new Queue<bool>([false, true]);
        int asked = 0;
        var install = new FeedInstallViewModel("alpha.q8", "Alpha", "1.0.0", false, rig.All.Feed, rig.All.Dispatcher, (_, _) => { },
            _ =>
            {
                asked++;
                return Task.FromResult(answers.Dequeue());
            });

        await install.StartAsync();

        Assert.True(install.IsCancelled);
        Assert.False(install.IsDone);
        Assert.Contains("not accepted", install.Status, StringComparison.Ordinal);
        Assert.Empty(rig.Store.List());

        await install.RetryCommand.ExecuteAsync(null);
        Assert.True(install.IsDone, install.Error);
        Assert.Equal(2, asked);
    }

    [Fact]
    public async Task FeedInstall_WithRestrictiveTerms_WaitsOnTheGate_ThroughTheCatalog()
    {
        using var rig = new FeedCatalogTests.Rig();
        await FeedCatalogTests.PublishAlphaAsync(rig);
        await rig.Catalog.LoadAsync();
        var package = rig.Catalog.Entries[0];
        var terms = LicenseNotices.TermsFor("feed-test", "Alpha", new PackageLicense { Spdx = "other", Gated = true });
        var install = new FeedInstallViewModel(package.Id, package.Name, package.Version, false, rig.All.Feed, rig.All.Dispatcher, (_, _) => { },
            token => rig.All.Licenses.RequestAsync(terms, token));

        var running = install.StartAsync();
        Assert.False(running.IsCompleted);
        Assert.Equal("feed:feed-test", rig.All.Licenses.Pending!.Key);
        rig.All.Licenses.Accept();
        await running;

        Assert.True(install.IsDone, install.Error);
    }

    [Fact]
    public void AcceptedTerms_HideThePostDownloadNotice()
    {
        using var services = new TestServices();
        var notice = new LicenseNotice("omni", "Non-commercial use only.", ["feed:omni.q8", "hub:owner/Omni-GGUF"]);
        Assert.False(LicenseNotices.IsHandled(services.Services, notice));

        services.Settings.Update(settings => settings with { AcceptedLicenses = [new AcceptedLicense("feed:omni.q8", "other", "v1", DateTimeOffset.UtcNow)] });

        Assert.True(LicenseNotices.IsHandled(services.Services, notice));
        Assert.False(LicenseNotices.IsHandled(services.Services, notice with { AcceptedKeys = ["feed:another"] }));
    }

    [Theory]
    [InlineData("light", 1200)]
    [InlineData("dark", 1200)]
    [InlineData("light", 600)]
    [InlineData("dark", 600)]
    public Task Dialog_ShowsTheTerms_AndAcceptAndCancelAnswer(string theme, int width) => HeadlessApp.RunAsync(async () =>
    {
        var sink = new BindingErrorSink();
        Logger.Sink = sink;
        using var services = new TestServices(dispatcher: new AvaloniaDispatcher());
        new ThemeService(Avalonia.Application.Current!).Apply(theme == "dark" ? AppTheme.Dark : AppTheme.Light, "Violet");
        var viewModel = new MainWindowViewModel(services.Services);
        var window = new MainWindow { DataContext = viewModel, Width = width, Height = 700 };
        window.Show();
        try
        {
            var dialog = window.FindControl<Panel>("LicenseDialog")!;
            Assert.False(dialog.IsVisible);

            var first = services.Licenses.RequestAsync(Terms());
            await HeadlessApp.RenderAsync(window);
            Assert.True(dialog.IsVisible);
            Assert.Equal("Accept the licence terms of Omni", window.FindControl<TextBlock>("LicenseTitle")!.Text);
            Assert.Equal("Licence: other", window.FindControl<TextBlock>("LicenseName")!.Text);
            Assert.Equal("Non-commercial use only.", window.FindControl<TextBlock>("LicenseNoticeText")!.Text);
            Assert.True(window.FindControl<Button>("LicenseLinkButton")!.IsVisible);
            var problems = LayoutChecker.Find(window, visual => visual is not TextBlock).Distinct().Take(8).ToList();
            string folder = Environment.GetEnvironmentVariable("SHARPLIGHT_STUDIO_SCREENSHOTS") is { Length: > 0 } configured ? configured : Path.Combine(AppContext.BaseDirectory, "page-screenshots");
            Directory.CreateDirectory(folder);
            using (var frame = window.CaptureRenderedFrame())
            {
                frame!.Save(Path.Combine(folder, $"studio-page-license-dialog-{theme}-{width}.png"), new PngBitmapEncoderOptions());
            }

            Assert.True(problems.Count == 0, string.Join("\n", problems));

            viewModel.LicenseDialog.OpenLicenseCommand.Execute(null);
            Assert.Equal("https://example.test/license", Assert.Single(services.Platform.Opened));

            viewModel.LicenseDialog.AcceptCommand.Execute(null);
            Assert.True(await first);
            await HeadlessApp.RenderAsync(window);
            Assert.False(dialog.IsVisible);

            var second = services.Licenses.RequestAsync(Terms("x"));
            await HeadlessApp.RenderAsync(window);
            Assert.True(dialog.IsVisible);
            viewModel.LicenseDialog.CancelCommand.Execute(null);
            Assert.False(await second);
            await HeadlessApp.RenderAsync(window);
            Assert.False(dialog.IsVisible);
            Assert.Empty(sink.Messages);
        }
        finally
        {
            window.Close();
            Dispatcher.UIThread.RunJobs();
            Logger.Sink = null;
        }
    });
}
