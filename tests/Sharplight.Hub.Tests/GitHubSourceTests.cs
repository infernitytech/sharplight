using System.Net;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using Sharplight.CatalogTool;
using Sharplight.Hub.Catalog;
using Sharplight.Package;

namespace Sharplight.Hub.Tests;

/// <summary>Catalog file sets from GitHub (publishers that release only there): addresses, validation, resolution and installation.</summary>
public sealed class GitHubSourceTests : IDisposable
{
    private const string Repository = "publisher/detector";
    private const string Commit = "be95df9152c0d7618fa1edfeb296fc3dae32376f";
    private const string FilePath = "src/data/model.onnx";
    private const string GitOid = "625ad8909b1abdba2292f3a9a10db6864fd5d561";
    private static readonly byte[] Content = Encoding.UTF8.GetBytes("onnx model bytes");
    private static readonly string ContentSha = Convert.ToHexStringLower(SHA256.HashData(Content));

    private readonly FakeHub _hub = new();
    private readonly TempDirectory _directory = new();
    private readonly HttpClient _http;

    public GitHubSourceTests()
    {
        _http = _hub.CreateClient();
        _hub.Intercept = ServeGitHub;
    }

    public void Dispose()
    {
        _http.Dispose();
        _hub.Dispose();
        _directory.Dispose();
    }

    [Fact]
    public void Addresses_ArePinnedToTheCommitOrTheRelease()
    {
        var files = Variant();
        var assets = Variant() with { Source = CatalogSources.GitHubRelease, Release = "v6.2" };
        var hub = CatalogSamples.Variant();

        Assert.Equal($"https://raw.githubusercontent.com/{Repository}/{Commit}/src/data/model.onnx", files.GetDirectFileUri(FilePath)!.AbsoluteUri);
        Assert.Equal($"https://github.com/{Repository}/releases/download/v6.2/model.onnx", assets.GetDirectFileUri("model.onnx")!.AbsoluteUri);
        Assert.Null(hub.GetDirectFileUri("Model-Q8_0.gguf"));
        Assert.Equal($"https://github.com/{Repository}/tree/{Commit}", files.GetPageUri(FakeHub.Endpoint).AbsoluteUri);
        Assert.Equal($"https://github.com/{Repository}/releases/tag/v6.2", assets.GetPageUri(FakeHub.Endpoint).AbsoluteUri);
        Assert.Equal($"https://hub.test/owner/Model-GGUF/tree/{CatalogSamples.Commit}", hub.GetPageUri(FakeHub.Endpoint).AbsoluteUri);
        Assert.False(files.IsHubSource);
        Assert.True(hub.IsHubSource);
    }

    [Fact]
    public void Validator_ChecksSourceReleaseAndChecksums()
    {
        Assert.Empty(CatalogValidator.Validate(CatalogSamples.Catalog(CatalogSamples.Entry("detector", Variant()))));

        var unknown = Variant() with { Source = "gitlab" };
        var noTag = Variant() with { Source = CatalogSources.GitHubRelease, Release = null };
        var tagOnHub = CatalogSamples.Variant() with { Release = "v1" };
        var noSha = Variant() with { Files = [new CatalogFile { Path = FilePath, Size = Content.Length, GitOid = GitOid }] };

        Assert.Contains(Problems(unknown), problem => problem.Contains("unknown source 'gitlab'", StringComparison.Ordinal));
        Assert.Contains(Problems(noTag), problem => problem.Contains("needs the release tag", StringComparison.Ordinal));
        Assert.Contains(Problems(tagOnHub), problem => problem.Contains("release tag needs a GitHub source", StringComparison.Ordinal));
        Assert.Contains(Problems(noSha), problem => problem.Contains("needs a sha256 (files from GitHub)", StringComparison.Ordinal));
    }

    [Fact]
    public async Task Install_DownloadsFromGitHubWithoutTheHubToken()
    {
        var store = new ModelStore(_directory.Path);
        var entry = CatalogSamples.Entry("detector", Variant()) with { Kind = "voice-activity" };
        var options = _hub.Options(token: "hf_secret");
        byte[]? converted = null;
        var puller = new ModelPuller(new HubClient(_http, options), new FileDownloader(_http, options), store, TestHost.Converter)
        {
            FreeSpace = _ => 1L << 40,
            CompositeConverter = (primary, _, destination, _) =>
            {
                converted = File.ReadAllBytes(primary);
                File.WriteAllText(destination, "package");
                return new ConversionResult(destination, "detector", "silero_vad", "onnx", 1, 1, 7, TimeSpan.Zero, TimeSpan.Zero, 0, 0, "silero_vad");
            },
        };

        var result = await puller.InstallAsync(CatalogInstallPlan.For(entry, entry.Variants[0]), cancellationToken: TestContext.Current.CancellationToken);

        var request = Assert.Single(_hub.Requests);
        Assert.Equal($"https://raw.githubusercontent.com/{Repository}/{Commit}/{FilePath}", request.RequestUri!.AbsoluteUri);
        Assert.Null(request.Headers.Authorization);
        Assert.Equal(Content, converted);
        Assert.Equal(Commit, result.Manifest.Commit);
        Assert.Equal(Repository, result.Manifest.Repository);
    }

    [Fact]
    public async Task Install_RejectsAFileThatDiffersFromItsChecksum()
    {
        var store = new ModelStore(_directory.Path);
        var variant = Variant() with { Files = [new CatalogFile { Path = FilePath, Size = Content.Length, Sha256 = CatalogSamples.Sha }] };
        var entry = CatalogSamples.Entry("detector", variant);
        var options = _hub.Options();
        var puller = new ModelPuller(new HubClient(_http, options), new FileDownloader(_http, options), store, TestHost.Converter) { FreeSpace = _ => 1L << 40 };

        await Assert.ThrowsAsync<HubException>(() => puller.InstallAsync(CatalogInstallPlan.For(entry, variant), cancellationToken: TestContext.Current.CancellationToken));

        Assert.Empty(store.List());
    }

    [Fact]
    public async Task Resolve_PinsTheReleaseTagAndHashesTheChosenFiles()
    {
        var unresolved = new CatalogVariant
        {
            Id = "onnx", Label = "ONNX", Format = CatalogFormats.Onnx, Repository = Repository, Source = CatalogSources.GitHub, Release = "v6.2",
            Include = ["src/data/*.onnx"],
        };
        var catalog = CatalogSamples.Catalog(CatalogSamples.Entry("detector", unresolved) with { License = "mit" });
        var github = new GitHubReader(_http, new Uri("https://api.github.test/"), token: null);
        var resolver = new CatalogResolver(new HubClient(_http, _hub.Options()), _ => { }, github);

        var resolved = (await resolver.ResolveAsync(catalog, ResolveMode.Missing, null, TestContext.Current.CancellationToken)).Entries[0].Variants[0];

        Assert.Empty(resolver.Failures);
        Assert.Equal(Commit, resolved.Revision);
        Assert.Equal(FilePath, resolved.Primary);
        var file = Assert.Single(resolved.Files);
        Assert.Equal((FilePath, (long)Content.Length, ContentSha, GitOid), (file.Path, file.Size, file.Sha256, file.GitOid));
        Assert.DoesNotContain(_hub.Requests, request => request.RequestUri!.Host == FakeHub.Endpoint.Host);
        Assert.Empty(CatalogValidator.Validate(CatalogSamples.Catalog(CatalogSamples.Entry("detector", resolved))));

        var checker = new CatalogChecker(new HubClient(_http, _hub.Options()), _ => { }, github);
        Assert.Empty(await checker.CheckAsync(CatalogSamples.Catalog(CatalogSamples.Entry("detector", resolved)), TestContext.Current.CancellationToken));
        var changed = resolved with { Files = [file with { GitOid = new string('1', 40) }] };
        Assert.Single(await checker.CheckAsync(CatalogSamples.Catalog(CatalogSamples.Entry("detector", changed)), TestContext.Current.CancellationToken));
    }

    private static CatalogVariant Variant() => new()
    {
        Id = "onnx", Label = "ONNX", Format = CatalogFormats.Onnx, Repository = Repository, Revision = Commit, Source = CatalogSources.GitHub,
        Release = "v6.2", Include = [FilePath], Primary = FilePath,
        Files = [new CatalogFile { Path = FilePath, Size = Content.Length, Sha256 = ContentSha, GitOid = GitOid }],
    };

    private static IReadOnlyList<string> Problems(CatalogVariant variant) =>
        CatalogValidator.Validate(CatalogSamples.Catalog(CatalogSamples.Entry("detector", variant)));

    /// <summary>The GitHub API (tag, tree) and raw file host for <see cref="Repository"/>; everything else goes to the fake Hub.</summary>
    private static HttpResponseMessage? ServeGitHub(HttpRequestMessage request)
    {
        var uri = request.RequestUri!;
        string? json = uri.Host switch
        {
            "api.github.test" when uri.AbsolutePath == $"/repos/{Repository}/commits/v6.2" => new JsonObject { ["sha"] = Commit }.ToJsonString(),
            "api.github.test" when uri.AbsolutePath == $"/repos/{Repository}/git/trees/{Commit}" => new JsonObject
            {
                ["truncated"] = false,
                ["tree"] = new JsonArray(
                    new JsonObject { ["path"] = "src", ["type"] = "tree", ["sha"] = new string('2', 40) },
                    new JsonObject { ["path"] = FilePath, ["type"] = "blob", ["size"] = Content.Length, ["sha"] = GitOid },
                    new JsonObject { ["path"] = "README.md", ["type"] = "blob", ["size"] = 5, ["sha"] = new string('3', 40) }),
            }.ToJsonString(),
            _ => null,
        };
        if (json is not null)
        {
            return new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(json, Encoding.UTF8, "application/json") };
        }

        if (uri.Host == "raw.githubusercontent.com" && uri.AbsolutePath == $"/{Repository}/{Commit}/{FilePath}")
        {
            return new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(Content) };
        }

        return uri.Host == FakeHub.Endpoint.Host ? null : new HttpResponseMessage(HttpStatusCode.NotFound);
    }
}
