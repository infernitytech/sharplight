using Sharplight.Studio.Services.Platform;
using Sharplight.Studio.Services.Threading;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>
/// Reports progress on the UI thread, throttled: reports from any thread are collected and handed to the UI in batches, at most about 15
/// times a second (<see cref="UpdateThrottle"/>), so a producer that reports per chunk or per kernel cannot flood the dispatcher. A
/// report that only refines the previous one (<c>replaces</c>) takes its place in the batch. Call <see cref="Flush"/> when the work is
/// done so the last reports are applied before the result is shown.
/// </summary>
/// <typeparam name="T">The progress value.</typeparam>
internal sealed class DispatchingProgress<T> : IProgress<T>, IDisposable
{
    private readonly IUiDispatcher _dispatcher;
    private readonly Action<IReadOnlyList<T>> _handler;
    private readonly Func<T, T, bool>? _replaces;
    private readonly UpdateThrottle _throttle;
    private readonly Lock _lock = new();
    private List<T> _pending = [];

    /// <param name="dispatcher">Runs the handler on the UI thread.</param>
    /// <param name="handler">Applies one report (on the UI thread, in report order).</param>
    /// <param name="replaces">Whether a report (second argument) supersedes the pending one before it (first); default never.</param>
    /// <param name="interval">The least time between two batches; default <see cref="UpdateThrottle.DefaultInterval"/>.</param>
    public DispatchingProgress(IUiDispatcher dispatcher, Action<T> handler, Func<T, T, bool>? replaces = null, TimeSpan? interval = null)
        : this(dispatcher, batch => Each(batch, handler), interval, replaces)
    {
    }

    /// <param name="dispatcher">Runs the handler on the UI thread.</param>
    /// <param name="batchHandler">Applies the reports collected since the last batch (on the UI thread, in report order).</param>
    /// <param name="interval">The least time between two batches; default <see cref="UpdateThrottle.DefaultInterval"/>.</param>
    /// <param name="replaces">Whether a report (second argument) supersedes the pending one before it (first); default never.</param>
    public DispatchingProgress(IUiDispatcher dispatcher, Action<IReadOnlyList<T>> batchHandler, TimeSpan? interval = null, Func<T, T, bool>? replaces = null)
    {
        _dispatcher = dispatcher;
        _handler = batchHandler;
        _replaces = replaces;
        _throttle = new UpdateThrottle(() => _dispatcher.Post(Apply), interval);
    }

    /// <inheritdoc/>
    public void Report(T value)
    {
        lock (_lock)
        {
            if (_replaces is not null && _pending.Count > 0 && _replaces(_pending[^1], value))
            {
                _pending[^1] = value;
            }
            else
            {
                _pending.Add(value);
            }
        }

        _throttle.Request();
    }

    /// <summary>Applies the reports still waiting (inline when called on the UI thread).</summary>
    public void Flush() => _dispatcher.Post(Apply);

    /// <inheritdoc/>
    public void Dispose() => _throttle.Dispose();

    private void Apply()
    {
        List<T> batch;
        lock (_lock)
        {
            if (_pending.Count == 0)
            {
                return;
            }

            batch = _pending;
            _pending = [];
        }

        _handler(batch);
    }

    private static void Each(IReadOnlyList<T> batch, Action<T> handler)
    {
        foreach (var value in batch)
        {
            handler(value);
        }
    }
}
