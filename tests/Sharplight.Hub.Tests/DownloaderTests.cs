using System.Net;

namespace Sharplight.Hub.Tests;

public sealed class DownloaderTests : IDisposable
{
    private const string RepositoryId = "owner/model";
    private const string FileName = "weights.bin";
    private const int FileSize = 300_000;

    private readonly FakeHub _hub = new();
    private readonly FakeRepository _repository;
    private readonly TempDirectory _directory = new();
    private readonly HttpClient _http;

    public DownloaderTests()
    {
        _repository = _hub.Add(RepositoryId).WithSize(FileName, FileSize).With("config.json", """{"a":1}""");
        _http = _hub.CreateClient();
    }

    public void Dispose()
    {
        _http.Dispose();
        _hub.Dispose();
        _directory.Dispose();
    }

    [Fact]
    public async Task DownloadsAndVerifiesLfsFile()
    {
        var destination = await DownloadAsync(FileName);

        Assert.Equal(_repository.Files[FileName], await File.ReadAllBytesAsync(destination, TestContext.Current.CancellationToken));
        Assert.False(File.Exists(destination + DownloadRequest.PartialExtension));
    }

    [Fact]
    public async Task VerifiesGitBlobIdOfSmallFiles()
    {
        var destination = await DownloadAsync("config.json");

        Assert.Equal("""{"a":1}""", await File.ReadAllTextAsync(destination, TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task ResumesPartialFileWithRangeRequest()
    {
        const int Half = FileSize / 2;
        var destination = _directory.Combine(FileName);
        await File.WriteAllBytesAsync(destination + DownloadRequest.PartialExtension, _repository.Files[FileName][..Half], TestContext.Current.CancellationToken);

        await DownloadAsync(FileName);

        var request = Assert.Single(_hub.Requests);
        Assert.Equal(Half, request.Headers.Range!.Ranges.Single().From);
        Assert.Equal(_repository.Files[FileName], await File.ReadAllBytesAsync(destination, TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task RestartsWhenServerIgnoresRange()
    {
        var destination = _directory.Combine(FileName);
        await File.WriteAllBytesAsync(destination + DownloadRequest.PartialExtension, new byte[1000], TestContext.Current.CancellationToken);
        _hub.IgnoreRanges = true;

        await DownloadAsync(FileName);

        Assert.Equal(_repository.Files[FileName], await File.ReadAllBytesAsync(destination, TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task RetriesTransientStatusWithBackoff()
    {
        int failures = 2;
        _hub.Intercept = request => IsFile(request) && failures-- > 0 ? new HttpResponseMessage(HttpStatusCode.ServiceUnavailable) : null;

        await DownloadAsync(FileName);

        Assert.Equal(3, _hub.Requests.Count);
    }

    [Fact]
    public async Task ResumesAfterConnectionDropsMidStream()
    {
        const int FailAfter = 100_000;
        bool failed = false;
        _hub.WrapBody = (_, body) =>
        {
            if (failed)
            {
                return body;
            }

            failed = true;
            return new FailingStream(body, FailAfter);
        };

        var destination = await DownloadAsync(FileName);

        Assert.Equal(2, _hub.Requests.Count);
        Assert.Null(_hub.Requests[0].Headers.Range);
        Assert.Equal(FailAfter, _hub.Requests[1].Headers.Range!.Ranges.Single().From);
        Assert.Equal(_repository.Files[FileName], await File.ReadAllBytesAsync(destination, TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task GivesUpAfterMaxAttemptsWithoutProgress()
    {
        _hub.Intercept = request => IsFile(request) ? new HttpResponseMessage(HttpStatusCode.BadGateway) : null;

        var error = await Assert.ThrowsAsync<HubException>(() => DownloadAsync(FileName));

        Assert.Contains("failed after 4 attempts", error.Message, StringComparison.Ordinal);
        Assert.Equal(4, _hub.Requests.Count);
    }

    [Fact]
    public async Task ChecksumMismatchDeletesPartialFile()
    {
        _repository.ReportedSha256[FileName] = new string('0', 64);
        var destination = _directory.Combine(FileName);

        var error = await Assert.ThrowsAsync<HubException>(() => DownloadAsync(FileName));

        Assert.Contains("checksum mismatch", error.Message, StringComparison.Ordinal);
        Assert.False(File.Exists(destination));
        Assert.False(File.Exists(destination + DownloadRequest.PartialExtension));
    }

    [Fact]
    public async Task NotFoundIsNotRetried()
    {
        var error = await Assert.ThrowsAsync<HubException>(() => DownloadAsync("missing.bin", size: 10));

        Assert.Equal(HttpStatusCode.NotFound, error.StatusCode);
        Assert.Single(_hub.Requests);
    }

    [Fact]
    public async Task GatedRepositoryExplainsToken()
    {
        _hub.Intercept = request =>
        {
            if (!IsFile(request))
            {
                return null;
            }

            var response = new HttpResponseMessage(HttpStatusCode.Unauthorized);
            response.Headers.Add("X-Error-Code", "GatedRepo");
            return response;
        };

        var error = await Assert.ThrowsAsync<HubException>(() => DownloadAsync(FileName));

        Assert.Contains("gated", error.Message, StringComparison.Ordinal);
        Assert.Contains(HubClientOptions.TokenVariable, error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task SendsTokenToHubOnly()
    {
        var options = _hub.Options(token: "secret-token");
        var downloader = new FileDownloader(_http, options);
        var hubFile = new DownloadRequest(new Uri(FakeHub.Endpoint, $"{RepositoryId}/resolve/{_repository.Commit}/{FileName}"), _directory.Combine("a"), FileSize);
        var otherHost = new DownloadRequest(new Uri("https://cdn.test/file"), _directory.Combine("b"), FileSize);
        _hub.Intercept = request => request.RequestUri!.Host == "cdn.test" ? new HttpResponseMessage(HttpStatusCode.NotFound) : null;

        await downloader.DownloadAsync(hubFile, cancellationToken: TestContext.Current.CancellationToken);
        await Assert.ThrowsAsync<HubException>(() => downloader.DownloadAsync(otherHost, cancellationToken: TestContext.Current.CancellationToken));

        Assert.Equal("secret-token", _hub.Requests[0].Headers.Authorization?.Parameter);
        Assert.Null(_hub.Requests[1].Headers.Authorization);
    }

    [Fact]
    public async Task ReportsProgressUpToTheFileSize()
    {
        var reports = new List<DownloadProgress>();
        var progress = new SynchronousProgress(reports.Add);

        await DownloadAsync(FileName, progress: progress);

        Assert.NotEmpty(reports);
        Assert.Equal(FileSize, reports[^1].Received);
        Assert.Equal(FileSize, reports[^1].Total);
    }

    private static bool IsFile(HttpRequestMessage request) => request.RequestUri!.AbsolutePath.Contains("/resolve/", StringComparison.Ordinal);

    private async Task<string> DownloadAsync(string path, long? size = null, IProgress<DownloadProgress>? progress = null)
    {
        var downloader = new FileDownloader(_http, _hub.Options());
        var client = new HubClient(_http, _hub.Options());
        var resolved = await client.GetRepositoryAsync(RepositoryId, cancellationToken: TestContext.Current.CancellationToken);
        _hub.Requests.Clear();
        var file = resolved.Files.FirstOrDefault(file => file.Path == path) ?? new HubFile(path, size ?? 0, null, null);
        var destination = _directory.Combine(path);
        await downloader.DownloadAsync(DownloadRequest.For(client, resolved, file, destination), progress, cancellationToken: TestContext.Current.CancellationToken);
        return destination;
    }
}
