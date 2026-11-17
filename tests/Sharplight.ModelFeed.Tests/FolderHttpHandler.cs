using System.Collections.Concurrent;
using System.Net;
using System.Net.Http.Headers;

namespace Sharplight.ModelFeed.Tests;

/// <summary>Serves a feed folder over fake HTTP under <c>/models/v1/</c>, with range requests.</summary>
internal sealed class FolderHttpHandler(string folder) : HttpMessageHandler
{
    public ConcurrentBag<long> Ranges { get; } = [];

    protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        var relative = request.RequestUri!.AbsolutePath["/models/v1/".Length..];
        var path = Path.Combine(folder, relative);
        if (!File.Exists(path))
        {
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound));
        }

        var bytes = File.ReadAllBytes(path);
        if (request.Headers.Range?.Ranges.FirstOrDefault()?.From is { } from)
        {
            Ranges.Add(from);
            var partial = new HttpResponseMessage(HttpStatusCode.PartialContent) { Content = new ByteArrayContent(bytes, (int)from, bytes.Length - (int)from) };
            partial.Content.Headers.ContentRange = new ContentRangeHeaderValue(from, bytes.Length - 1, bytes.Length);
            return Task.FromResult(partial);
        }

        return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(bytes) });
    }
}
