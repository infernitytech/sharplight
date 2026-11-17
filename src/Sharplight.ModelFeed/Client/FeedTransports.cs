namespace Sharplight.ModelFeed.Client;

/// <summary>Creates the transport for a feed location.</summary>
public static class FeedTransports
{
    /// <summary>A transport for <paramref name="location"/>: <c>http(s)://</c> URLs use <paramref name="http"/>, <c>file://</c> URLs and paths a folder.</summary>
    /// <param name="location">Feed URL or folder.</param>
    /// <param name="http">HTTP client for HTTP feeds (owned by the caller).</param>
    /// <returns>The transport.</returns>
    /// <exception cref="ArgumentException">The location is neither an HTTP(S) URL nor a folder.</exception>
    public static IFeedTransport Create(string location, HttpClient http)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(location);
        ArgumentNullException.ThrowIfNull(http);
        if (Uri.TryCreate(location, UriKind.Absolute, out var uri))
        {
            if (uri.Scheme == Uri.UriSchemeHttp || uri.Scheme == Uri.UriSchemeHttps)
            {
                return new HttpFeedTransport(http, uri);
            }

            if (uri.IsFile)
            {
                return new FolderFeedTransport(uri.LocalPath);
            }

            throw new ArgumentException($"Feed scheme '{uri.Scheme}' is not supported (http, https, file or a folder path).", nameof(location));
        }

        return new FolderFeedTransport(location);
    }
}
