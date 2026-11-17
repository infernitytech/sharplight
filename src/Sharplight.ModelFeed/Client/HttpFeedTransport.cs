using System.Net;
using System.Net.Http.Headers;

namespace Sharplight.ModelFeed.Client;

/// <summary>A feed on an HTTP(S) server or object storage; package files are read with range requests. Thread safe.</summary>
public sealed class HttpFeedTransport : IFeedTransport
{
    private readonly HttpClient _http;
    private readonly Uri _base;

    /// <summary>Creates the transport.</summary>
    /// <param name="http">HTTP client (owned by the caller).</param>
    /// <param name="baseUri">The feed's base URI (a trailing slash is added).</param>
    public HttpFeedTransport(HttpClient http, Uri baseUri)
    {
        ArgumentNullException.ThrowIfNull(http);
        ArgumentNullException.ThrowIfNull(baseUri);
        _http = http;
        _base = baseUri.AbsoluteUri.EndsWith('/') ? baseUri : new Uri(baseUri.AbsoluteUri + "/");
    }

    /// <inheritdoc />
    public string Location => _base.AbsoluteUri;

    /// <inheritdoc />
    public async Task<byte[]?> ReadAsync(string path, long maxBytes, CancellationToken cancellationToken = default)
    {
        using var request = new HttpRequestMessage(HttpMethod.Get, new Uri(_base, path));
        request.Headers.CacheControl = new CacheControlHeaderValue { NoCache = true };
        using var response = await SendAsync(request, cancellationToken).ConfigureAwait(false);
        if (response.StatusCode == HttpStatusCode.NotFound)
        {
            return null;
        }

        Ensure(response, path);
        if (response.Content.Headers.ContentLength is { } length && length > maxBytes)
        {
            throw new IOException($"{path} is larger than the allowed {maxBytes} bytes.");
        }

        await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
        using var buffer = new MemoryStream();
        var chunk = new byte[81920];
        int read;
        while ((read = await stream.ReadAsync(chunk, cancellationToken).ConfigureAwait(false)) > 0)
        {
            if (buffer.Length + read > maxBytes)
            {
                throw new IOException($"{path} is larger than the allowed {maxBytes} bytes.");
            }

            buffer.Write(chunk, 0, read);
        }

        return buffer.ToArray();
    }

    /// <inheritdoc />
    public async Task<Stream> OpenReadAsync(string path, long offset, CancellationToken cancellationToken = default)
    {
        var request = new HttpRequestMessage(HttpMethod.Get, new Uri(_base, path));
        if (offset > 0)
        {
            request.Headers.Range = new RangeHeaderValue(offset, null);
        }

        var response = await SendAsync(request, cancellationToken).ConfigureAwait(false);
        try
        {
            Ensure(response, path);
            if (offset > 0 && response.StatusCode != HttpStatusCode.PartialContent)
            {
                throw new IOException($"{Location}{path} does not support range requests.");
            }

            return new ResponseStream(await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false), response, request);
        }
        catch
        {
            response.Dispose();
            request.Dispose();
            throw;
        }
    }

    private async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        try
        {
            return await _http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
        }
        catch (HttpRequestException exception)
        {
            throw new IOException($"{request.RequestUri} could not be read: {exception.Message}", exception);
        }
    }

    private void Ensure(HttpResponseMessage response, string path)
    {
        if (!response.IsSuccessStatusCode)
        {
            throw new IOException($"{Location}{path} answered {(int)response.StatusCode} {response.ReasonPhrase}.");
        }
    }
}
