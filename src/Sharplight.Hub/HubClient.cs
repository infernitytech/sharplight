using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace Sharplight.Hub;

/// <summary>
/// Reads model metadata from the Hugging Face Hub: search, a repository's commit and file list (with sizes and checksums), and
/// small files. Transient failures are retried (<see cref="HubClientOptions.Retry"/>); other failures throw
/// <see cref="HubException"/> with a message that says what to do. Thread safe.
/// </summary>
public sealed partial class HubClient
{
    private static readonly string[] s_searchExpansions = ["downloads", "likes", "gated", "library_name", "pipeline_tag", "gguf", "safetensors", "siblings"];

    private readonly HttpClient _http;
    private readonly HubClientOptions _options;

    /// <summary>Creates a client on <paramref name="http"/> (see <see cref="HubClientOptions.CreateHttpClient"/>), which the caller owns.</summary>
    /// <param name="http">The HTTP client to send requests with.</param>
    /// <param name="options">Endpoint, token and retry settings, or <see langword="null"/> for the public Hub without a token.</param>
    public HubClient(HttpClient http, HubClientOptions? options = null)
    {
        _http = http;
        _options = options ?? new HubClientOptions();
    }

    /// <summary>The client's options.</summary>
    public HubClientOptions Options => _options;

    /// <summary>Searches model repositories, most downloaded first.</summary>
    /// <param name="query">What to search for.</param>
    /// <param name="cancellationToken">Cancels the request.</param>
    /// <returns>Up to <see cref="HubSearchQuery.Limit"/> results.</returns>
    public async Task<IReadOnlyList<HubModelSummary>> SearchAsync(HubSearchQuery query, CancellationToken cancellationToken = default)
    {
        var url = new StringBuilder($"api/models?search={Uri.EscapeDataString(query.Text)}&sort=downloads&direction=-1");
        url.Append(CultureInfo.InvariantCulture, $"&limit={Math.Max(1, query.Limit)}");
        if (query.Format is { } format)
        {
            url.Append(CultureInfo.InvariantCulture, $"&filter={FormatFilter(format)}");
        }

        foreach (var expansion in s_searchExpansions)
        {
            url.Append(CultureInfo.InvariantCulture, $"&expand%5B%5D={expansion}");
        }

        var json = await GetStringAsync(new Uri(_options.Endpoint, url.ToString()), $"Search '{query.Text}'", "", cancellationToken).ConfigureAwait(false);
        return HubJson.ParseSearch(json);
    }

    /// <summary>Resolves <paramref name="revision"/> of a repository to its commit and lists every file at that commit.</summary>
    /// <param name="repository">The repository id, <c>owner/name</c>.</param>
    /// <param name="revision">Branch, tag or commit (default <c>main</c>).</param>
    /// <param name="cancellationToken">Cancels the requests.</param>
    /// <exception cref="HubException">The repository or revision does not exist or is not accessible.</exception>
    public async Task<HubRepository> GetRepositoryAsync(string repository, string revision = "main", CancellationToken cancellationToken = default)
    {
        var infoUri = new Uri(_options.Endpoint, $"api/models/{repository}/revision/{Uri.EscapeDataString(revision)}");
        var info = HubJson.ParseRevision(await GetStringAsync(infoUri, repository, repository, cancellationToken).ConfigureAwait(false));
        var id = info.Id ?? repository;

        var files = new List<HubFile>();
        Uri? page = new(_options.Endpoint, $"api/models/{id}/tree/{info.Commit}?recursive=true");
        while (page is not null)
        {
            var (json, next) = await GetPageAsync(page, id, id, cancellationToken).ConfigureAwait(false);
            files.AddRange(HubJson.ParseTree(json));
            page = next;
        }

        return new HubRepository(id, revision, info.Commit, info.Gated, files);
    }

    /// <summary>Downloads a small text file (e.g. a safetensors index) of a resolved repository.</summary>
    /// <param name="repository">The resolved repository.</param>
    /// <param name="path">The file's path in the repository.</param>
    /// <param name="cancellationToken">Cancels the request.</param>
    public Task<string> GetFileTextAsync(HubRepository repository, string path, CancellationToken cancellationToken = default)
    {
        return GetStringAsync(GetFileUri(repository, path), path, repository.Id, cancellationToken);
    }

    /// <summary>The download address of a file, pinned to the repository's resolved commit.</summary>
    /// <param name="repository">The resolved repository.</param>
    /// <param name="path">The file's path in the repository.</param>
    public Uri GetFileUri(HubRepository repository, string path)
    {
        var escaped = string.Join('/', path.Split('/').Select(Uri.EscapeDataString));
        return new Uri(_options.Endpoint, $"{repository.Id}/resolve/{repository.Commit}/{escaped}");
    }

    private static string FormatFilter(ModelFormat format) => format switch
    {
        ModelFormat.Gguf => "gguf",
        ModelFormat.Safetensors => "safetensors",
        _ => "onnx",
    };

    private async Task<string> GetStringAsync(Uri uri, string what, string repository, CancellationToken cancellationToken)
    {
        return (await GetPageAsync(uri, what, repository, cancellationToken).ConfigureAwait(false)).Json;
    }

    /// <summary>GETs <paramref name="uri"/> with retries; returns the body and the <c>Link: rel="next"</c> page, if any.</summary>
    private async Task<(string Json, Uri? Next)> GetPageAsync(Uri uri, string what, string repository, CancellationToken cancellationToken)
    {
        for (int attempt = 1; ; attempt++)
        {
            using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            timeout.CancelAfter(_options.RequestTimeout);
            try
            {
                using var request = HubRequests.Get(uri, _options);
                using var response = await _http.SendAsync(request, HttpCompletionOption.ResponseContentRead, timeout.Token).ConfigureAwait(false);
                if (!response.IsSuccessStatusCode)
                {
                    throw HubRequests.Failure(response, what, repository, _options);
                }

                var body = await response.Content.ReadAsStringAsync(timeout.Token).ConfigureAwait(false);
                return (body, NextPage(response));
            }
            catch (Exception error) when (RetryPolicy.IsTransient(error, cancellationToken))
            {
                if (attempt >= _options.Retry.MaxAttempts)
                {
                    throw new HubException($"{what}: the Hub did not answer after {attempt} attempts ({RetryPolicy.Describe(error)}).", inner: error);
                }

                await Task.Delay(RetryPolicy.Delay(_options.Retry, attempt, error), cancellationToken).ConfigureAwait(false);
            }
        }
    }

    private Uri? NextPage(HttpResponseMessage response)
    {
        if (!response.Headers.TryGetValues("Link", out var links))
        {
            return null;
        }

        foreach (var link in links)
        {
            var match = NextLink().Match(link);
            if (match.Success)
            {
                return new Uri(_options.Endpoint, match.Groups["uri"].Value);
            }
        }

        return null;
    }

    [GeneratedRegex(@"<(?<uri>[^>]+)>\s*;\s*rel=""?next""?")]
    private static partial Regex NextLink();
}
