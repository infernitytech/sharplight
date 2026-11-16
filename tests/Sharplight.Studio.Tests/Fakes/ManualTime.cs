namespace Sharplight.Studio.Tests.Fakes;

/// <summary>A clock with timers that moves only when a test calls <see cref="Advance"/>; due timers fire on the caller's thread.</summary>
internal sealed class ManualTime : TimeProvider
{
    private readonly List<ManualTimer> _timers = [];
    private long _ticks;

    /// <inheritdoc/>
    public override long TimestampFrequency => TimeSpan.TicksPerSecond;

    /// <inheritdoc/>
    public override long GetTimestamp() => _ticks;

    /// <inheritdoc/>
    public override ITimer CreateTimer(TimerCallback callback, object? state, TimeSpan dueTime, TimeSpan period)
    {
        var timer = new ManualTimer(this, callback, state);
        timer.Change(dueTime, period);
        return timer;
    }

    /// <summary>Moves the clock and fires the timers that became due.</summary>
    /// <param name="by">How far.</param>
    public void Advance(TimeSpan by)
    {
        _ticks += by.Ticks;
        foreach (var timer in _timers.Where(timer => timer.DueTicks <= _ticks).ToList())
        {
            timer.Fire();
        }
    }

    private sealed class ManualTimer(ManualTime time, TimerCallback callback, object? state) : ITimer
    {
        public long DueTicks { get; private set; } = long.MaxValue;

        public bool Change(TimeSpan dueTime, TimeSpan period)
        {
            time._timers.Remove(this);
            if (dueTime != Timeout.InfiniteTimeSpan)
            {
                DueTicks = time._ticks + dueTime.Ticks;
                time._timers.Add(this);
            }

            return true;
        }

        public void Fire()
        {
            time._timers.Remove(this);
            callback(state);
        }

        public void Dispose() => time._timers.Remove(this);

        public ValueTask DisposeAsync()
        {
            Dispose();
            return ValueTask.CompletedTask;
        }
    }
}
