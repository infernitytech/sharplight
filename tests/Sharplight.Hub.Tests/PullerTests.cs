using System.Net;
using Sharplight.Package.Import;

namespace Sharplight.Hub.Tests;

public sealed class PullerTests : IDisposable
{
    private const string RepositoryId = "owner/Model-GGUF";
    private const long PlentyOfSpace = 1L << 40;

    private readonly FakeHub _hub = new();
    private readonly TempDirectory _directory = new();
    private readonly HttpClient _http;
    private readonly ModelStore _store;
    private readonly List<(string Primary, string? Projector)> _conversions = [];

    public PullerTests()
    {
        var repository = _hub.Add(RepositoryId);
        repository.Architecture = "llama";
        repository
            .WithSize("Model-Q4_K_M.gguf", 20_000)
            .WithSize("Model-Q8_0.gguf", 40_000)
            .WithSize("mmproj-F16.gguf", 5_000)
            .With("README.md", "readme");
        _http = _hub.CreateClient();
        _store = new ModelStore(_directory.Path);
    }

    public void Dispose()
    {
        _http.Dispose();
        _hub.Dispose();
        _directory.Dispose();
    }

    [Fact]
    public async Task PullDownloadsConvertsRegistersAndDeletesSources()
    {
        var result = await CreatePuller().PullAsync(ModelReference.Parse(RepositoryId), cancellationToken: TestContext.Current.CancellationToken);

        var manifest = result.Manifest;
        Assert.False(result.UpToDate);
        Assert.Equal(25_000, result.DownloadedBytes);
        Assert.Equal("Q4_K_M", manifest.Tag);
        Assert.Equal("llama", manifest.Architecture);
        Assert.Equal("model:q4_k_m", manifest.ShortName);
        Assert.Equal(["Model-Q4_K_M.gguf", "mmproj-F16.gguf"], manifest.SourceFiles.Select(file => file.Path));
        Assert.True(File.Exists(manifest.PackagePath));
        Assert.False(Directory.Exists(Path.Combine(manifest.Directory, ModelStore.SourceDirectoryName)));
        Assert.False(File.Exists(Path.Combine(manifest.Directory, ModelStore.PendingPackageFileName)));
        var (primary, projector) = Assert.Single(_conversions);
        Assert.EndsWith(Path.Combine("Model-GGUF", "Model-Q4_K_M.gguf"), primary, StringComparison.Ordinal);
        Assert.EndsWith("mmproj-F16.gguf", projector, StringComparison.Ordinal);
        Assert.Equal(manifest.Name, _store.Find("model")?.Name);
    }

    [Fact]
    public async Task KeepSourceKeepsDownloadedFiles()
    {
        var options = new PullOptions { KeepSource = true };

        var result = await CreatePuller().PullAsync(ModelReference.Parse(RepositoryId + ":Q8_0"), options, TestContext.Current.CancellationToken);

        Assert.True(result.Manifest.SourceKept);
        Assert.True(File.Exists(Path.Combine(result.Manifest.Directory, ModelStore.SourceDirectoryName, "Model-GGUF", "Model-Q8_0.gguf")));
    }

    [Fact]
    public async Task SecondPullAtSameCommitIsUpToDate()
    {
        var puller = CreatePuller();
        await puller.PullAsync(ModelReference.Parse(RepositoryId), cancellationToken: TestContext.Current.CancellationToken);
        _hub.Requests.Clear();

        var again = await puller.PullAsync(ModelReference.Parse(RepositoryId), cancellationToken: TestContext.Current.CancellationToken);

        Assert.True(again.UpToDate);
        Assert.Single(_conversions);
        Assert.DoesNotContain(_hub.Requests, request => request.RequestUri!.AbsolutePath.Contains("/resolve/", StringComparison.Ordinal));
    }

    [Fact]
    public async Task FailedConversionLeavesNothingRegisteredAndResumes()
    {
        var failing = CreatePuller(convert: (_, _, _, _) => throw new HubException("conversion failed (test)"));

        await Assert.ThrowsAsync<HubException>(() => failing.PullAsync(ModelReference.Parse(RepositoryId), cancellationToken: TestContext.Current.CancellationToken));

        Assert.Empty(_store.List());
        _hub.Requests.Clear();
        var result = await CreatePuller().PullAsync(ModelReference.Parse(RepositoryId), cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal(0, result.DownloadedBytes);
        Assert.Equal(0, _hub.CountRequests(".gguf"));
        Assert.Single(_store.List());
    }

    [Fact]
    public async Task InterruptedDownloadIsNotRegisteredAndResumesFromPartialFile()
    {
        _hub.Intercept = request => request.RequestUri!.AbsolutePath.EndsWith("mmproj-F16.gguf", StringComparison.Ordinal)
            ? new HttpResponseMessage(HttpStatusCode.Forbidden)
            : null;
        await Assert.ThrowsAsync<HubException>(() => CreatePuller().PullAsync(ModelReference.Parse(RepositoryId), cancellationToken: TestContext.Current.CancellationToken));
        Assert.Empty(_store.List());

        _hub.Intercept = null;
        var result = await CreatePuller().PullAsync(ModelReference.Parse(RepositoryId), cancellationToken: TestContext.Current.CancellationToken);

        Assert.Equal(5_000, result.DownloadedBytes);
        Assert.Single(_store.List());
    }

    [Fact]
    public async Task InsufficientDiskSpaceFailsBeforeDownloading()
    {
        var puller = CreatePuller(freeSpace: 10_000);

        var error = await Assert.ThrowsAsync<HubException>(() => puller.PullAsync(ModelReference.Parse(RepositoryId), cancellationToken: TestContext.Current.CancellationToken));

        Assert.Contains("Not enough disk space", error.Message, StringComparison.Ordinal);
        Assert.Equal(0, _hub.CountRequests(".gguf"));
    }

    [Fact]
    public async Task GatedRepositoryWithoutTokenFailsEarly()
    {
        _hub.Add("meta/Gated", gated: true).WithSize("m-Q8_0.gguf", 2000);

        var error = await Assert.ThrowsAsync<HubException>(() => CreatePuller().PullAsync(ModelReference.Parse("meta/Gated"), cancellationToken: TestContext.Current.CancellationToken));

        Assert.Contains(HubClientOptions.TokenVariable, error.Message, StringComparison.Ordinal);
    }

    private ModelPuller CreatePuller(Func<string, string?, string, Action<string>?, ConversionResult>? convert = null, long freeSpace = PlentyOfSpace)
    {
        var options = _hub.Options();
        return new ModelPuller(new HubClient(_http, options), new FileDownloader(_http, options), _store)
        {
            FreeSpace = _ => freeSpace,
            Converter = convert ?? FakeConvert,
        };
    }

    private ConversionResult FakeConvert(string primary, string? projector, string destination, Action<string>? log)
    {
        Assert.True(File.Exists(primary));
        _conversions.Add((primary, projector));
        File.WriteAllText(destination, "package");
        return new ConversionResult(destination, "Model", "decoder", "gguf", 1, 1, 7, TimeSpan.Zero, TimeSpan.Zero, 0, 0, "qwen3");
    }
}
