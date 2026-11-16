using Avalonia.Media.Imaging;
using Sharplight.Studio.Services.Platform;

namespace Sharplight.Studio.ViewModels.Gallery;

/// <summary>
/// Thumbnails of the gallery: decoded from the PNG files on worker threads (two at a time, so a long list does not occupy every
/// core), kept for the most recently shown few hundred. Reading <see cref="Get"/> never blocks: it returns what is cached and starts
/// the decoding of what is not.
/// </summary>
internal sealed class ThumbnailCache
{
    /// <summary>The decode width of a thumbnail.</summary>
    public const int Width = 192;

    private const int Capacity = 400;

    private readonly IUiDispatcher _dispatcher;
    private readonly object _lock = new();
    private readonly Dictionary<string, LinkedListNode<(string Path, Bitmap? Bitmap)>> _cached = [];
    private readonly LinkedList<(string Path, Bitmap? Bitmap)> _recent = new();
    private readonly HashSet<string> _loading = [];
    private readonly SemaphoreSlim _workers = new(2);

    /// <param name="dispatcher">Raises the change of an item's thumbnail on the UI thread.</param>
    public ThumbnailCache(IUiDispatcher dispatcher) => _dispatcher = dispatcher;

    /// <summary>The cached thumbnail of <paramref name="path"/>, or null while it loads (<paramref name="item"/> is notified then).</summary>
    public Bitmap? Get(GalleryItemViewModel item, string path)
    {
        lock (_lock)
        {
            if (_cached.TryGetValue(path, out var node))
            {
                _recent.Remove(node);
                _recent.AddFirst(node);
                return node.Value.Bitmap;
            }

            if (!_loading.Add(path))
            {
                return null;
            }
        }

        _ = Task.Run(async () =>
        {
            await _workers.WaitAsync().ConfigureAwait(false);
            Bitmap? bitmap = null;
            try
            {
                using var stream = File.OpenRead(path);
                bitmap = Bitmap.DecodeToWidth(stream, Width);
            }
            catch (Exception error) when (error is IOException or UnauthorizedAccessException or ArgumentException or InvalidOperationException or NotSupportedException)
            {
                // No thumbnail (no Avalonia platform in tests, or the file is gone).
            }
            finally
            {
                _workers.Release();
            }

            lock (_lock)
            {
                _loading.Remove(path);
                _cached[path] = _recent.AddFirst((path, bitmap));
                while (_recent.Count > Capacity)
                {
                    _cached.Remove(_recent.Last!.Value.Path);
                    _recent.RemoveLast();
                }
            }

            if (bitmap is not null)
            {
                _dispatcher.Post(item.ThumbnailReady);
            }
        });
        return null;
    }

    /// <summary>Forgets a thumbnail (its image was deleted).</summary>
    public void Remove(string path)
    {
        lock (_lock)
        {
            if (_cached.Remove(path, out var node))
            {
                _recent.Remove(node);
            }
        }
    }
}
