namespace Sharplight.Studio.Services.Threading;

/// <summary>
/// Coalesces requests from any thread into at most one call of an action per interval: the first request after a quiet interval fires
/// at once (on the requesting thread), later ones within the interval fire once at its end (on a timer thread), so the last state is
/// always delivered. Used to keep progress, previews and streamed text from flooding the UI thread.
/// </summary>
internal sealed class UpdateThrottle : IDisposable
{
    /// <summary>The default interval: about 15 updates per second.</summary>
    public static readonly TimeSpan DefaultInterval = TimeSpan.FromMilliseconds(66);

    private readonly Action _fire;
    private readonly TimeSpan _interval;
    private readonly TimeProvider _time;
    private readonly Lock _lock = new();
    private long _lastFired;
    private bool _hasFired;
    private bool _pending;
    private ITimer? _timer;
    private bool _disposed;

    /// <param name="fire">What a request triggers; called on the requesting thread or a timer thread, never concurrently with itself.</param>
    /// <param name="interval">The least time between two calls; default <see cref="DefaultInterval"/>.</param>
    /// <param name="time">The clock (tests); default the system clock.</param>
    public UpdateThrottle(Action fire, TimeSpan? interval = null, TimeProvider? time = null)
    {
        _fire = fire;
        _interval = interval ?? DefaultInterval;
        _time = time ?? TimeProvider.System;
    }

    /// <summary>Asks for a call: now if the last one is at least an interval ago, else once at the end of the interval.</summary>
    public void Request()
    {
        lock (_lock)
        {
            if (_disposed)
            {
                return;
            }

            if (_timer is not null)
            {
                _pending = true;
                return;
            }

            var elapsed = _hasFired ? _time.GetElapsedTime(_lastFired) : TimeSpan.MaxValue;
            if (elapsed < _interval)
            {
                _pending = true;
                _timer = _time.CreateTimer(_ => OnTimer(), null, _interval - elapsed, Timeout.InfiniteTimeSpan);
                return;
            }

            MarkFired();
            _fire();
        }
    }

    /// <summary>Makes a pending call now (on the calling thread), e.g. before a final state is shown.</summary>
    public void Flush()
    {
        lock (_lock)
        {
            if (!_pending || _disposed)
            {
                return;
            }

            StopTimer();
            MarkFired();
            _fire();
        }
    }

    /// <summary>Drops a pending call and stops the timer.</summary>
    public void Dispose()
    {
        lock (_lock)
        {
            _disposed = true;
            _pending = false;
            StopTimer();
        }
    }

    private void OnTimer()
    {
        lock (_lock)
        {
            StopTimer();
            if (!_pending || _disposed)
            {
                return;
            }

            MarkFired();
            _fire();
        }
    }

    private void MarkFired()
    {
        _pending = false;
        _hasFired = true;
        _lastFired = _time.GetTimestamp();
    }

    private void StopTimer()
    {
        _timer?.Dispose();
        _timer = null;
    }
}
