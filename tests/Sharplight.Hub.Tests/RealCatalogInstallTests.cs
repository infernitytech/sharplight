using Sharplight.CatalogTool;
using Sharplight.Hub.Catalog;
using Sharplight.Package;

namespace Sharplight.Hub.Tests;

/// <summary>
/// Installs a tiny entry of the committed catalog from the real Hub, end to end (pinned download, checksum, conversion, registration);
/// runs only when <c>SHARPLIGHT_HUB_NETWORK_TESTS=1</c> (downloads about 31 MB).
/// </summary>
public sealed class RealCatalogInstallTests
{
    private const string EnableVariable = "SHARPLIGHT_HUB_NETWORK_TESTS";

    [Fact]
    public async Task InstallsTheSmallestWhisperFromTheCatalog()
    {
        Assert.SkipUnless(Environment.GetEnvironmentVariable(EnableVariable) == "1", $"set {EnableVariable}=1 to run network tests");
        var root = CatalogPaths.FindRepositoryRoot(AppContext.BaseDirectory);
        var catalog = VerifiedCatalogSerializer.Parse(File.ReadAllText(Path.Combine(root, CatalogPaths.Catalog)));
        var entry = catalog.Find("whisper-tiny")!;
        var plan = CatalogInstallPlan.For(entry, entry.Variants.Single(variant => variant.Id == "q5-1-whispercpp"));
        using var directory = new TempDirectory();
        using var http = HubClientOptions.CreateHttpClient();
        var options = HubClientOptions.FromEnvironment();
        var puller = new ModelPuller(new HubClient(http, options), new FileDownloader(http, options), new ModelStore(directory.Path), TestHost.Converter);

        var result = await puller.InstallAsync(plan, cancellationToken: TestContext.Current.CancellationToken);
        var again = await puller.InstallAsync(plan, cancellationToken: TestContext.Current.CancellationToken);

        Assert.Equal("whisper-tiny/q5-1-whispercpp", result.Manifest.Catalog);
        Assert.Equal(plan.Variant.Revision, result.Manifest.Commit);
        Assert.NotEmpty(ModelPackageReader.Open(result.Manifest.PackagePath).Definition.Graphs);
        Assert.True(again.UpToDate);
    }
}
