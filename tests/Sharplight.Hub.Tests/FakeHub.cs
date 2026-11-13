using System.Net;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json.Nodes;

namespace Sharplight.Hub.Tests;

/// <summary>
/// An in-memory Hub behind an <see cref="HttpMessageHandler"/>: serves revision info, tree listings and file contents (with
/// byte ranges) for the repositories added to it. <see cref="Intercept"/> injects failures; every request is recorded.
/// </summary>
internal sealed class FakeHub : HttpMessageHandler
{
    public static readonly Uri Endpoint = new("https://hub.test/");

    private readonly Dictionary<string, FakeRepository> _repositories = new(StringComparer.Ordinal);

    /// <summary>Returns a response to send instead of the normal one, or null to serve normally.</summary>
    public Func<HttpRequestMessage, HttpResponseMessage?>? Intercept { get; set; }

    /// <summary>Wraps a served file body (e.g. to fail after some bytes); receives the request and the body.</summary>
    public Func<HttpRequestMessage, Stream, Stream>? WrapBody { get; set; }

    /// <summary>Serve the full body with 200 even for range requests.</summary>
    public bool IgnoreRanges { get; set; }

    public List<HttpRequestMessage> Requests { get; } = [];

    public FakeRepository Add(string id, string commit = "0123456789abcdef0123456789abcdef01234567", bool gated = false)
    {
        var repository = new FakeRepository(id, commit, gated);
        _repositories[id] = repository;
        return repository;
    }

    public HubClientOptions Options(string? token = null) => new()
    {
        Endpoint = Endpoint,
        Token = token,
        RequestTimeout = TimeSpan.FromSeconds(10),
        StallTimeout = TimeSpan.FromSeconds(10),
        Retry = new RetryOptions { MaxAttempts = 4, InitialDelay = TimeSpan.FromMilliseconds(1), MaxDelay = TimeSpan.FromMilliseconds(5) },
    };

    public HttpClient CreateClient() => new(this, disposeHandler: false);

    public int CountRequests(string pathSuffix) => Requests.Count(request => request.RequestUri!.AbsolutePath.EndsWith(pathSuffix, StringComparison.Ordinal));

    protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        lock (Requests)
        {
            Requests.Add(request);
        }

        var response = Intercept?.Invoke(request) ?? Serve(request);
        response.RequestMessage = request;
        return Task.FromResult(response);
    }

    private HttpResponseMessage Serve(HttpRequestMessage request)
    {
        var path = Uri.UnescapeDataString(request.RequestUri!.AbsolutePath.TrimStart('/'));
        if (path.StartsWith("api/models/", StringComparison.Ordinal))
        {
            return ServeApi(path["api/models/".Length..]);
        }

        var parts = path.Split('/', 5);
        if (parts.Length == 5 && parts[2] == "resolve" && _repositories.TryGetValue($"{parts[0]}/{parts[1]}", out var repository)
            && parts[3] == repository.Commit && repository.Files.TryGetValue(parts[4], out var content))
        {
            return ServeFile(request, content);
        }

        return NotFound("EntryNotFound");
    }

    private HttpResponseMessage ServeApi(string rest)
    {
        foreach (var repository in _repositories.Values)
        {
            if (rest.StartsWith(repository.Id + "/revision/", StringComparison.Ordinal))
            {
                var info = new JsonObject { ["id"] = repository.Id, ["sha"] = repository.Commit, ["gated"] = repository.Gated }.ToJsonString();
                return Json(info);
            }

            if (rest == $"{repository.Id}/tree/{repository.Commit}")
            {
                return Json(repository.TreeJson());
            }
        }

        return NotFound("RepoNotFound");
    }

    private HttpResponseMessage ServeFile(HttpRequestMessage request, byte[] content)
    {
        long from = request.Headers.Range?.Ranges.First().From ?? 0;
        bool partial = from > 0 && !IgnoreRanges;
        if (partial && from >= content.Length)
        {
            return new HttpResponseMessage(HttpStatusCode.RequestedRangeNotSatisfiable);
        }

        var body = partial ? content[(int)from..] : content;
        Stream stream = new MemoryStream(body);
        if (WrapBody is not null)
        {
            stream = WrapBody(request, stream);
        }

        var response = new HttpResponseMessage(partial ? HttpStatusCode.PartialContent : HttpStatusCode.OK) { Content = new StreamContent(stream) };
        response.Content.Headers.ContentLength = body.Length;
        if (partial)
        {
            response.Content.Headers.ContentRange = new ContentRangeHeaderValue(from, content.Length - 1, content.Length);
        }

        return response;
    }

    private static HttpResponseMessage Json(string json) => new(HttpStatusCode.OK) { Content = new StringContent(json, Encoding.UTF8, "application/json") };

    private static HttpResponseMessage NotFound(string code)
    {
        var response = new HttpResponseMessage(HttpStatusCode.NotFound);
        response.Headers.Add("X-Error-Code", code);
        return response;
    }
}
