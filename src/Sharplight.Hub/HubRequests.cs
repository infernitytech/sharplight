using System.Net;
using System.Net.Http.Headers;

namespace Sharplight.Hub;

/// <summary>Builds Hub requests (User-Agent, token) and turns failed responses into user-facing errors.</summary>
internal static class HubRequests
{
    private const string ErrorCodeHeader = "X-Error-Code";
    private const string ErrorMessageHeader = "X-Error-Message";
    private const string GatedRepoCode = "GatedRepo";
    private const string RepoNotFoundCode = "RepoNotFound";
    private const string RevisionNotFoundCode = "RevisionNotFound";
    private const string EntryNotFoundCode = "EntryNotFound";

    /// <summary>A GET request; the token is only attached for the Hub's own host.</summary>
    public static HttpRequestMessage Get(Uri uri, HubClientOptions options)
    {
        var request = new HttpRequestMessage(HttpMethod.Get, uri);
        request.Headers.UserAgent.ParseAdd(options.UserAgent);
        if (options.Token is { } token && string.Equals(uri.Host, options.Endpoint.Host, StringComparison.OrdinalIgnoreCase))
        {
            request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
        }

        return request;
    }

    /// <summary>
    /// The error for a failed response: a <see cref="TransientHttpException"/> for statuses worth retrying, otherwise a
    /// <see cref="HubException"/> that says what to do (authenticate, accept the terms, check the name).
    /// </summary>
    /// <param name="response">The failed response.</param>
    /// <param name="what">What was requested, for the message (e.g. <c>Qwen/Qwen3-0.6B</c> or a file name).</param>
    /// <param name="repository">The repository, for the link to its page.</param>
    /// <param name="options">The client options (whether a token was sent).</param>
    public static Exception Failure(HttpResponseMessage response, string what, string repository, HubClientOptions options)
    {
        var status = response.StatusCode;
        if (RetryPolicy.IsTransient(status))
        {
            var retryAfter = response.Headers.RetryAfter is { } header
                ? header.Delta ?? (header.Date - DateTimeOffset.UtcNow)
                : null;
            return new TransientHttpException(status, retryAfter > TimeSpan.Zero ? retryAfter : null, $"{what}: HTTP {(int)status} {response.ReasonPhrase}");
        }

        var code = Header(response, ErrorCodeHeader);
        var serverMessage = Header(response, ErrorMessageHeader);
        bool hasToken = options.Token is not null;
        var page = new Uri(options.Endpoint, repository);
        string message = (status, code) switch
        {
            (_, GatedRepoCode) => hasToken
                ? $"{what}: {repository} is a gated repository and the token in {HubClientOptions.TokenVariable} has no access yet. Accept its terms on {page} and try again."
                : $"{what}: {repository} is a gated repository. Accept its terms on {page}, create an access token and set {HubClientOptions.TokenVariable}.",
            (_, RepoNotFoundCode) or (HttpStatusCode.Unauthorized, null) when !hasToken =>
                $"{what}: repository {repository} not found. Check the name; private repositories need {HubClientOptions.TokenVariable}.",
            (_, RevisionNotFoundCode) => $"{what}: revision not found in {repository}.",
            (_, EntryNotFoundCode) or (HttpStatusCode.NotFound, _) => $"{what}: not found in {repository} (HTTP 404).",
            (HttpStatusCode.Unauthorized, _) => $"{what}: the token in {HubClientOptions.TokenVariable} was rejected (HTTP 401) or has no access to {repository}.",
            (HttpStatusCode.Forbidden, _) => $"{what}: access to {repository} denied (HTTP 403). If it is gated, accept its terms on {page}; check that {HubClientOptions.TokenVariable} grants read access.",
            _ => $"{what}: HTTP {(int)status} {response.ReasonPhrase}" + (serverMessage is not null ? $" ({serverMessage})" : ""),
        };
        return new HubException(message, status);
    }

    private static string? Header(HttpResponseMessage response, string name)
    {
        return response.Headers.TryGetValues(name, out var values) ? values.FirstOrDefault() : null;
    }
}
