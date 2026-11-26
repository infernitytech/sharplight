using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Text.Json.Nodes;
using Sharplight.Hub;
using Sharplight.Hub.Catalog;

namespace Sharplight.CatalogTool;

/// <summary>
/// Reads catalog file sets that come from GitHub (<see cref="CatalogSources.GitHub"/>, <see cref="CatalogSources.GitHubRelease"/>) through
/// the GitHub REST API: the commit a tag or branch resolves to, the repository files at a commit (path, size, git blob id) and a release's
/// assets (name, size, sha256 digest). Repository files have no sha256 in the API, so <see cref="Sha256Async"/> downloads them.
/// <c>GITHUB_TOKEN</c>, when set, is sent to the API only (higher rate limit); public files need no token.
/// </summary>
/// <param name="http">The client (redirects followed).</param>
/// <param name="api">The API address (<see cref="CatalogSources.GitHubApi"/>).</param>
/// <param name="token">A GitHub token for the API, or <see langword="null"/>.</param>
internal sealed class GitHubReader(HttpClient http, Uri api, string? token)
{
    /// <summary>The environment variable with an optional GitHub token.</summary>
    public const string TokenVariable = "GITHUB_TOKEN";

    private const string UserAgent = "sharplight-catalog-tool";

    /// <summary>A reader of the public GitHub API with the token from <see cref="TokenVariable"/>, if any.</summary>
    /// <param name="http">The client.</param>
    public static GitHubReader FromEnvironment(HttpClient http) =>
        new(http, CatalogSources.GitHubApi, Environment.GetEnvironmentVariable(TokenVariable) is { Length: > 0 } value ? value : null);

    /// <summary>The commit a tag, branch or commit of <paramref name="repository"/> resolves to.</summary>
    /// <param name="repository"><c>owner/name</c>.</param>
    /// <param name="gitRef">A tag, branch or commit.</param>
    /// <param name="cancellationToken">Cancels the request.</param>
    /// <exception cref="HubException">The repository or reference does not exist, or the API failed.</exception>
    public async Task<string> ResolveCommitAsync(string repository, string gitRef, CancellationToken cancellationToken)
    {
        var json = await GetJsonAsync($"repos/{repository}/commits/{Uri.EscapeDataString(gitRef)}", cancellationToken).ConfigureAwait(false);
        return json["sha"]?.GetValue<string>() ?? throw new HubException($"{repository}@{gitRef}: GitHub returned no commit.");
    }

    /// <summary>Every file of <paramref name="repository"/> at <paramref name="commit"/> with its size and git blob id (no sha256).</summary>
    /// <param name="repository"><c>owner/name</c>.</param>
    /// <param name="commit">A commit.</param>
    /// <param name="cancellationToken">Cancels the request.</param>
    /// <exception cref="HubException">The tree is not readable or was truncated by the API.</exception>
    public async Task<IReadOnlyList<HubFile>> ListFilesAsync(string repository, string commit, CancellationToken cancellationToken)
    {
        var json = await GetJsonAsync($"repos/{repository}/git/trees/{commit}?recursive=1", cancellationToken).ConfigureAwait(false);
        if (json["truncated"]?.GetValue<bool>() == true)
        {
            throw new HubException($"{repository}@{commit}: the repository tree is too large for one listing.");
        }

        var files = new List<HubFile>();
        foreach (var item in json["tree"]?.AsArray() ?? [])
        {
            if (item?["type"]?.GetValue<string>() == "blob")
            {
                files.Add(new HubFile(item["path"]!.GetValue<string>(), item["size"]?.GetValue<long>() ?? 0, null, item["sha"]?.GetValue<string>()));
            }
        }

        return files;
    }

    /// <summary>The assets of the release <paramref name="tag"/> with their sizes and sha256 digests (empty when GitHub has none).</summary>
    /// <param name="repository"><c>owner/name</c>.</param>
    /// <param name="tag">The release tag.</param>
    /// <param name="cancellationToken">Cancels the request.</param>
    /// <exception cref="HubException">The release does not exist, or the API failed.</exception>
    public async Task<IReadOnlyList<HubFile>> ListReleaseAssetsAsync(string repository, string tag, CancellationToken cancellationToken)
    {
        var json = await GetJsonAsync($"repos/{repository}/releases/tags/{Uri.EscapeDataString(tag)}", cancellationToken).ConfigureAwait(false);
        var files = new List<HubFile>();
        foreach (var asset in json["assets"]?.AsArray() ?? [])
        {
            string? digest = asset?["digest"]?.GetValue<string>();
            string? sha256 = digest is not null && digest.StartsWith("sha256:", StringComparison.Ordinal) ? digest["sha256:".Length..] : null;
            files.Add(new HubFile(asset!["name"]!.GetValue<string>(), asset["size"]?.GetValue<long>() ?? 0, sha256, null));
        }

        return files;
    }

    /// <summary>Downloads <paramref name="uri"/> and returns the lower-case hex SHA-256 of its content.</summary>
    /// <param name="uri">The file's address.</param>
    /// <param name="cancellationToken">Cancels the download.</param>
    /// <exception cref="HubException">The download failed.</exception>
    public async Task<string> Sha256Async(Uri uri, CancellationToken cancellationToken)
    {
        using var request = new HttpRequestMessage(HttpMethod.Get, uri);
        request.Headers.UserAgent.ParseAdd(UserAgent);
        using var response = await http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
        if (!response.IsSuccessStatusCode)
        {
            throw new HubException($"{uri}: HTTP {(int)response.StatusCode} {response.ReasonPhrase}");
        }

        await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
        return Convert.ToHexStringLower(await SHA256.HashDataAsync(stream, cancellationToken).ConfigureAwait(false));
    }

    private async Task<JsonNode> GetJsonAsync(string path, CancellationToken cancellationToken)
    {
        using var request = new HttpRequestMessage(HttpMethod.Get, new Uri(api, path));
        request.Headers.UserAgent.ParseAdd(UserAgent);
        request.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("application/vnd.github+json"));
        if (token is not null)
        {
            request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
        }

        using var response = await http.SendAsync(request, cancellationToken).ConfigureAwait(false);
        if (!response.IsSuccessStatusCode)
        {
            throw new HubException($"GitHub {path}: HTTP {(int)response.StatusCode} {response.ReasonPhrase}");
        }

        var body = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
        return JsonNode.Parse(body) ?? throw new HubException($"GitHub {path}: empty answer.");
    }
}
