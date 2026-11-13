namespace Sharplight.Server;

/// <summary>Streams an event source into an <see cref="SseWriter"/>, flushing only when no further event is ready.</summary>
internal static class SsePump
{
    /// <summary>
    /// Writes every item of <paramref name="source"/> with <paramref name="write"/> (which buffers into <paramref name="sse"/> and
    /// returns false to stop early). The socket is flushed whenever the next item is not available yet, so a burst of ready
    /// events (several tokens for a slow reader, a token plus the finish event) costs one flush instead of one per event.
    /// Whatever the last items wrote is still buffered on return: the caller writes its trailer and flushes.
    /// </summary>
    /// <returns>False when <paramref name="write"/> stopped the stream.</returns>
    public static async Task<bool> PumpAsync<T>(IAsyncEnumerable<T> source, SseWriter sse, Func<T, bool> write, CancellationToken ct)
    {
        await using var items = source.GetAsyncEnumerator(ct);
        while (true)
        {
            var next = items.MoveNextAsync();
            if (!next.IsCompleted)
            {
                await sse.FlushAsync();
            }

            if (!await next)
            {
                return true;
            }

            if (!write(items.Current))
            {
                return false;
            }
        }
    }
}
