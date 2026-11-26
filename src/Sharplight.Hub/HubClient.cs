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
    private static readonly string[] s_searchExpansions = ["downloads", "likes", "gated", "library_name", "pipeline_tag", "gguf", "safetensors", "siblings", "config", "createdAt", "lastModified", "tags"];

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
        if (query.Task is not { } task)
        {
            return await SearchOnceAsync(query, null, query.Text, query.Limit, cancellationToken).ConfigureAwait(false);
        }

        // The Hub filters one pipeline tag per request: a task with several tags is searched once per tag and merged.
        var tags = ModelTasks.SearchTags(task);
        var text = query.Text.Length == 0 ? ModelTasks.SearchHint(task) : query.Text;
        int limit = ModelTasks.IsRefinedByName(task) ? query.Limit * 2 : query.Limit;
        var requests = (tags.Count == 0 ? [null] : tags.Select(tag => (string?)tag)).Select(tag => SearchOnceAsync(query, tag, text, limit, cancellationToken)).ToList();
        var lists = await Task.WhenAll(requests).ConfigureAwait(false);
        var merged = Interleave(lists.Select(list => (IReadOnlyList<HubModelSummary>)[.. list.Where(model => model.Task == task)]).ToList())
            .DistinctBy(model => model.Id, StringComparer.OrdinalIgnoreCase);
        var ordered = query.Sort switch
        {
            HubSortOrder.Downloads => merged.OrderByDescending(model => model.Downloads),
            HubSortOrder.Likes => merged.OrderByDescending(model => model.Likes),
            HubSortOrder.Newest => merged.OrderByDescending(model => model.CreatedAt),
            HubSortOrder.Modified => merged.OrderByDescending(model => model.LastModified),
            _ => merged,
        };
        return [.. ordered.Take(Math.Max(1, query.Limit))];
    }

    private static IEnumerable<HubModelSummary> Interleave(IReadOnlyList<IReadOnlyList<HubModelSummary>> lists)
    {
        int longest = lists.Count == 0 ? 0 : lists.Max(list => list.Count);
        for (int i = 0; i < longest; i++)
        {
            foreach (var list in lists)
            {
                if (i < list.Count)
                {
                    yield return list[i];
                }
            }
        }
    }

    private async Task<IReadOnlyList<HubModelSummary>> SearchOnceAsync(HubSearchQuery query, string? pipeline, string text, int limit, CancellationToken cancellationToken)
    {
        var url = new StringBuilder($"api/models?sort={SortKey(query.Sort)}&direction=-1");
        if (text.Length > 0)
        {
            url.Append(CultureInfo.InvariantCulture, $"&search={Uri.EscapeDataString(text)}");
        }

        url.Append(CultureInfo.InvariantCulture, $"&limit={Math.Max(1, limit)}");
        if (pipeline is not null)
        {
            url.Append(CultureInfo.InvariantCulture, $"&pipeline_tag={Uri.EscapeDataString(pipeline)}");
        }

        if (query.Format is { } format && FormatFilter(format) is { } filter)
        {
            url.Append(CultureInfo.InvariantCulture, $"&filter={filter}");
        }

        if (query.License is { Length: > 0 } license)
        {
            url.Append(CultureInfo.InvariantCulture, $"&filter={Uri.EscapeDataString("license:" + license)}");
        }

        foreach (var expansion in s_searchExpansions)
        {
            url.Append(CultureInfo.InvariantCulture, $"&expand%5B%5D={expansion}");
        }

        string what = text.Length > 0 ? $"Search '{text}'" : "Model listing";
        var json = await GetStringAsync(new Uri(_options.Endpoint, url.ToString()), what, "", cancellationToken).ConfigureAwait(false);
        var results = HubJson.ParseSearch(json);
        return query.Format is { } wanted && FormatFilter(wanted) is null ? [.. results.Where(model => model.Formats.Contains(wanted))] : results;
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

        if (files.Any(file => file.Sha256 is null && file.GitOid is null))
        {
            // LFS checksums masked in the tree (a gated repository the token has no access to): the revision's blob listing has them, so
            // a catalog can pin the official files before access is granted.
            var blobsUri = new Uri(_options.Endpoint, $"api/models/{id}/revision/{info.Commit}?blobs=true");
            var hashes = HubJson.ParseLfsHashes(await GetStringAsync(blobsUri, id, id, cancellationToken).ConfigureAwait(false));
            for (int i = 0; i < files.Count; i++)
            {
                if (files[i] is { Sha256: null, GitOid: null } file && hashes.TryGetValue(file.Path, out var sha256))
                {
                    files[i] = file with { Sha256 = sha256 };
                }
            }
        }

        return new HubRepository(id, revision, info.Commit, info.Gated, files, info.Architecture, info.License, info.BaseModel);
    }

    /// <summary>Downloads a small text file (e.g. a safetensors index) of a resolved repository.</summary>
    /// <param name="repository">The resolved repository.</param>
    /// <param name="path">The file's path in the repository.</param>
    /// <param name="cancellationToken">Cancels the request.</param>
    public Task<string> GetFileTextAsync(HubRepository repository, string path, CancellationToken cancellationToken = default)
    {
        return GetStringAsync(GetFileUri(repository, path), path, repository.Id, cancellationToken);
    }

    /// <summary>
    /// Downloads the first bytes of a file (an HTTP range request), e.g. to read a model file's header before downloading it. A server
    /// that ignores the range sends the whole file; only <paramref name="count"/> bytes of it are read.
    /// </summary>
    /// <param name="repository">The resolved repository.</param>
    /// <param name="path">The file's path in the repository.</param>
    /// <param name="count">Bytes to read from the start of the file.</param>
    /// <param name="cancellationToken">Cancels the request.</param>
    /// <returns>Up to <paramref name="count"/> bytes (fewer when the file is shorter).</returns>
    public Task<byte[]> GetFilePrefixAsync(HubRepository repository, string path, int count, CancellationToken cancellationToken = default)
        => GetFileRangeAsync(repository, path, 0, count, cancellationToken);

    /// <summary>
    /// Downloads <paramref name="count"/> bytes of a file starting at <paramref name="offset"/> (an HTTP range request), e.g. to read a
    /// section of a model file that follows a part not worth downloading.
    /// </summary>
    /// <param name="repository">The resolved repository.</param>
    /// <param name="path">The file's path in the repository.</param>
    /// <param name="offset">The first byte to read.</param>
    /// <param name="count">Bytes to read.</param>
    /// <param name="cancellationToken">Cancels the request.</param>
    /// <returns>Up to <paramref name="count"/> bytes (fewer when the file ends earlier).</returns>
    /// <exception cref="HubException">The server ignored the range of a request that does not start at the beginning of the file.</exception>
    public async Task<byte[]> GetFileRangeAsync(HubRepository repository, string path, long offset, int count, CancellationToken cancellationToken = default)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(offset);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(count);
        var uri = GetFileUri(repository, path);
        for (int attempt = 1; ; attempt++)
        {
            using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            timeout.CancelAfter(_options.RequestTimeout);
            try
            {
                using var request = HubRequests.Get(uri, _options);
                request.Headers.Range = new System.Net.Http.Headers.RangeHeaderValue(offset, offset + count - 1);
                using var response = await _http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, timeout.Token).ConfigureAwait(false);
                if (response.StatusCode == System.Net.HttpStatusCode.RequestedRangeNotSatisfiable)
                {
                    return [];
                }

                if (!response.IsSuccessStatusCode)
                {
                    throw HubRequests.Failure(response, path, repository.Id, _options);
                }

                if (offset > 0 && response.StatusCode != System.Net.HttpStatusCode.PartialContent)
                {
                    throw new HubException($"{path}: the server does not answer range requests.");
                }

                await using var stream = await response.Content.ReadAsStreamAsync(timeout.Token).ConfigureAwait(false);
                var buffer = new byte[count];
                int read = await stream.ReadAtLeastAsync(buffer, count, throwOnEndOfStream: false, timeout.Token).ConfigureAwait(false);
                return read == count ? buffer : buffer[..read];
            }
            catch (Exception error) when (RetryPolicy.IsTransient(error, cancellationToken))
            {
                if (attempt >= _options.Retry.MaxAttempts)
                {
                    throw new HubException($"{path}: the Hub did not answer after {attempt} attempts ({RetryPolicy.Describe(error)}).", inner: error);
                }

                await Task.Delay(RetryPolicy.Delay(_options.Retry, attempt, error), cancellationToken).ConfigureAwait(false);
            }
        }
    }

    /// <summary>The download address of a file, pinned to the repository's resolved commit.</summary>
    /// <param name="repository">The resolved repository.</param>
    /// <param name="path">The file's path in the repository.</param>
    public Uri GetFileUri(HubRepository repository, string path)
    {
        var escaped = string.Join('/', path.Split('/').Select(Uri.EscapeDataString));
        return new Uri(_options.Endpoint, $"{repository.Id}/resolve/{repository.Commit}/{escaped}");
    }

    private static string SortKey(HubSortOrder sort) => sort switch
    {
        HubSortOrder.Likes => "likes",
        HubSortOrder.Trending => "trendingScore",
        HubSortOrder.Newest => "createdAt",
        HubSortOrder.Modified => "lastModified",
        _ => "downloads",
    };

    /// <summary>
    /// The Hub's filter for a format, or null when it has none: PyTorch checkpoints have no tag of their own (the <c>pytorch</c> library
    /// tag also marks safetensors repositories and is missing on many checkpoint folders), so their results are filtered by their files.
    /// </summary>
    private static string? FormatFilter(ModelFormat format) => format switch
    {
        ModelFormat.Gguf => "gguf",
        ModelFormat.Safetensors => "safetensors",
        ModelFormat.Diffusers => "diffusers",
        ModelFormat.WhisperCpp => "ggml",
        ModelFormat.Slm => "slm",
        ModelFormat.PyTorch => null,
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
