using Sharplight.Studio.Platform;
using Sharplight.Studio.Services.Platform;

namespace Sharplight.Studio.Tests;

/// <summary>The Avalonia dispatcher, counting the jobs posted to the UI thread (to check that runs do not flood it).</summary>
internal sealed class CountingDispatcher : IUiDispatcher
{
    private readonly AvaloniaDispatcher _inner = new();
    private int _posts;

    /// <summary>The jobs posted so far.</summary>
    public int Posts => Volatile.Read(ref _posts);

    /// <inheritdoc/>
    public void Post(Action action)
    {
        Interlocked.Increment(ref _posts);
        _inner.Post(action);
    }

    /// <summary>Starts counting anew.</summary>
    public void Reset() => Interlocked.Exchange(ref _posts, 0);
}
