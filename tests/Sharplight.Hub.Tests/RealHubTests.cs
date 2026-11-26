using Sharplight.Package;

namespace Sharplight.Hub.Tests;

/// <summary>
/// Talks to the real Hugging Face Hub; runs only when <c>SHARPLIGHT_HUB_NETWORK_TESTS=1</c> (downloads about 640 MB).
/// </summary>
public sealed class RealHubTests
{
    private const string EnableVariable = "SHARPLIGHT_HUB_NETWORK_TESTS";
    private const string SmallModel = "Qwen/Qwen3-0.6B-GGUF:Q8_0";

    private static bool Enabled => Environment.GetEnvironmentVariable(EnableVariable) == "1";

    [Fact]
    public async Task SearchFindsTheOfficialGguf()
    {
        Assert.SkipUnless(Enabled, $"set {EnableVariable}=1 to run network tests");
        using var http = HubClientOptions.CreateHttpClient();
        var client = new HubClient(http, HubClientOptions.FromEnvironment());

        var results = await client.SearchAsync(new HubSearchQuery { Text = "Qwen3-0.6B-GGUF", Format = ModelFormat.Gguf, Limit = 50 }, TestContext.Current.CancellationToken);

        var official = Assert.Single(results, result => result.Id == "Qwen/Qwen3-0.6B-GGUF");
        Assert.Contains("Q8_0", official.Variants);
    }

    [Fact]
    public async Task SearchFindsSeedVr2_AsPyTorch_AndSelectsTheFilesItsImporterNeeds()
    {
        Assert.SkipUnless(Enabled, $"set {EnableVariable}=1 to run network tests");
        using var http = HubClientOptions.CreateHttpClient();
        var client = new HubClient(http, HubClientOptions.FromEnvironment());

        var results = await client.SearchAsync(new HubSearchQuery { Text = "SeedVR2", Format = ModelFormat.PyTorch, Limit = 50 }, TestContext.Current.CancellationToken);
        var three = await client.GetRepositoryAsync("ByteDance-Seed/SeedVR2-3B", cancellationToken: TestContext.Current.CancellationToken);
        var seven = await client.GetRepositoryAsync("ByteDance-Seed/SeedVR2-7B", cancellationToken: TestContext.Current.CancellationToken);

        Assert.Contains(results, result => result.Id == "ByteDance-Seed/SeedVR2-3B");
        Assert.Contains(results, result => result.Id == "ByteDance-Seed/SeedVR2-7B");
        Assert.All(results, result => Assert.Contains(ModelFormat.PyTorch, result.Formats));
        Assert.Equal(["ema_vae.pth", "neg_emb.pt", "pos_emb.pt", "seedvr2_ema_3b.pth"], ModelFileSelector.Select(three, null).Files.Select(file => file.Path));
        Assert.Equal(["seedvr2_ema_7b", "seedvr2_ema_7b_sharp"], ModelFileSelector.ListVariants(seven, null));
    }

    [Fact]
    public async Task PullsConvertsAndStoresSmallGguf()
    {
        Assert.SkipUnless(Enabled, $"set {EnableVariable}=1 to run network tests");
        using var directory = new TempDirectory();
        using var http = HubClientOptions.CreateHttpClient();
        var options = HubClientOptions.FromEnvironment();
        var store = new ModelStore(directory.Path);
        var puller = new ModelPuller(new HubClient(http, options), new FileDownloader(http, options), store, TestHost.Converter);

        var result = await puller.PullAsync(ModelReference.Parse(SmallModel), cancellationToken: TestContext.Current.CancellationToken);

        var manifest = result.Manifest;
        Assert.Equal("Q8_0", manifest.Tag);
        Assert.Equal(40, manifest.Commit.Length);
        Assert.Equal(64, Assert.Single(manifest.SourceFiles).Sha256!.Length);
        Assert.False(Directory.Exists(Path.Combine(manifest.Directory, ModelStore.SourceDirectoryName)));
        var package = ModelPackageReader.Open(manifest.PackagePath);
        Assert.Contains(package.Definition.Graphs, graph => graph.Name == "decode");
        Assert.Equal(manifest.Name, store.Find("qwen3-0.6b:q8_0")?.Name);
    }
}
